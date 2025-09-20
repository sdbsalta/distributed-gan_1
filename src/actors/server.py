import math
from typing import List, Tuple
import torch.cuda
import torch.distributed as dist
import torch
from torch.futures import Future
import torch.utils.data
import logging
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import csv

def _compute_fid_score(
    real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute the Frechet Inception Distance
    :param real_images: The real images
    :param fake_images: The generated images
    :return: The Frechet Inception Distance
    """
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return FrechetInceptionDistance.compute(fid)


def _compute_inception_score(fake_images: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute the inception score
    :param fake_images: The generated images
    :return: The inception score
    """
    inception = InceptionScore(normalize=True, splits=1).to(device)
    inception.update(fake_images)
    return InceptionScore.compute(inception)[0]


def _split_dataset(
    dataset_size: int, world_size: int, iid: bool = False, generator: torch.Generator = torch.Generator(), device: torch.device = torch.device("cpu")
) -> List[torch.Tensor]:
    """
    Split the dataset into N parts, where N is the number of workers.
    Each worker will get a different part of the dataset.
    :param dataset_size: The size of the dataset
    :param world_size: The number of workers
    :param rank: The rank of the current worker
    :param iid: Whether to shuffle the dataset before splitting
    :return: A list of tensors, each tensor hold the indices of the dataset for each worker
    """
    if iid:
        indices = torch.randperm(dataset_size, device=device, generator=generator)
    else:
        indices = torch.arange(dataset_size, device=device)
    # Split the dataset into N parts
    split_indices = torch.chunk(indices, world_size)
    return split_indices


def start(
    backend: str,
    rank: int,
    generator_lr: float,
    world_size: int,
    batch_size: int,
    epochs: int,
    log_interval: int,
    generator: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    z_dim: int,
    log_folder: Path,
    image_shape: Tuple[int, int, int],
    dataset_name: str,
    device: torch.device = torch.device("cpu"),
    n_samples: int = 5,
    iid: bool = True,
    swap_interval: int = 1,
    beta_1: float = 0.5,
    beta_2: float = 0.999,
):
    # Initialize the process group (TCP)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(weeks=52),
    )
    dist.barrier()

    # Define the communication device
    communication_device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        communication_device = torch.device("cuda")

    logging.info(f"Server {rank} initialized, communication device: {communication_device}")
    print(generator)


    image_save_dir: Path = Path("saved_images")
    name = f"mdgan.{world_size-1}.{dataset_name}"
    logs_file: Path = log_folder / f"{name}.server.logs.csv"

    # Initialize the generator, notice that the server do not hold a loss function
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(beta_1, beta_2)
    )

    # N is the number of workers, world_size includes the server
    N = world_size - 1

    # K is the number of data batch the generator will generate for every epoch, many workers will therefore use the same data
    # since K < N
    k = max(math.floor(math.log(N)), 2)  # The results are reflects a version with the following line: X_d = K[n % k + 1]
    logging.info(f"Server {rank} has {N} workers and K={k}")

    # Determine the evaluation device
    evaluation_device = torch.device("cpu")
    if torch.cuda.is_available():
        evaluation_device = torch.device("cuda")

    # Create a random batch of real images to compute FID and IS scores
    # it can remain constant
    g = torch.Generator()
    g.manual_seed(0)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=n_samples, shuffle=True, generator=g
    )
    real_images: torch.Tensor = next(iter(data_loader))[0].to(device=evaluation_device)
    # If the images are grayscale, repeat the images to have 3 channels (torchmetrics requires 3 channels images)
    if real_images.shape[1] < 3:
        real_images = real_images.repeat(1, 3, 1, 1)
    # Normalize the images from 0 to 1 insatead of -1 to 1
    real_images = (real_images + 1) * 0.5
    grid_real = make_grid(
        real_images.to(dtype=torch.float32), nrow=4, normalize=True, value_range=(0, 1), padding=0
    )
    # Convert the grid to a PIL image
    grid_pil = to_pil_image(grid_real)
    # Create the save directory if it doesn't exist and save the grid
    Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    grid_path = Path(image_save_dir) / f"real_images.png"
    grid_pil.save(grid_path)

    # Split the dataset into N parts in a IID or non-IID manner
    device_generator = torch.Generator(device=communication_device)
    device_generator.manual_seed(0)
    list_of_indices = _split_dataset(len(dataset), world_size - 1, iid=iid, generator=device_generator, device=communication_device)
    logging.info(f"Server {rank} split the dataset into {len(list_of_indices)} parts")
    # Send the indices to the workers to inform them about the data they will use
    for n, indices in enumerate(list_of_indices):
        dist.send(
            tensor=torch.tensor(
                len(indices), device=communication_device, dtype=torch.int
            ),
            dst=n + 1,
        )
        dist.send(tensor=indices, dst=n + 1)
        logging.info(
            f"Server sent indices to worker {n + 1} with shape {indices.shape}"
        )

    # Create the save directory if it doesn't exist
    weights_path = Path("weights")
    generator.train()
    feedbacks = torch.zeros((N, batch_size, *image_shape), device=communication_device, requires_grad=True, dtype=torch.float32)
    size_feedback = feedbacks.element_size() * feedbacks.nelement() / 1024**2 # in MB

    fake_data = torch.zeros((2, batch_size, *image_shape), device=communication_device, dtype=torch.float32)
    size_fake_data = fake_data.element_size() * fake_data.nelement() / 1024**2 # in MB

    # Initialize the logs
    get_log = lambda epoch: {
        "epoch": epoch,
        "start.epoch": time.time(),
        "end.epoch": None,
        "start.epoch_calculation": time.time(),
        "end.epoch_calculation": None,
        "start.send_data": None,
        "end.send_data": None,
        "start.recv_data": None,
        "end.recv_data": None,
        "start.calc_gradients": None,
        "end.calc_gradients": None,
        "start.agg_gradients": None,
        "end.agg_gradients": None,
        "start.generate_data": None,
        "end.generate_data": None,
        "fid": None,
        "is": None,
        "start.fid": None,
        "end.fid": None,
        "start.is": None,
        "end.is": None,
        "size.data": size_fake_data,
        "size.feedback": size_feedback,
        "start.swap": None,
        "end.swap": None,
        "swap": False,
        "size.sent": 0,
        "size.recv": 0,
    }
    f = open(logs_file, "a", encoding="utf-8")
    csv_writer = csv.DictWriter(f, fieldnames=get_log(0).keys())
    csv_writer.writeheader()

    for epoch in range(epochs):
        logging.info(f"Server {rank} starting epoch {epoch}")
        current_logs = get_log(epoch)

        current_logs["start.generate_data"] = time.time()
        # Generate K batches of data
        seed = torch.randn((k * batch_size, z_dim, 1, 1), device=device)
        X: torch.tensor = generator(seed).to(device=device)
        logging.info(f"Server {rank} generated data with shape {X.shape}")
        # split in <k> batches
        K = torch.chunk(X, k)
        logging.info(f"Server {rank} generated {len(K)} batches of data, each with shape {K[0].shape}")
        current_logs["end.generate_data"] = time.time()

        current_logs["start.send_data"] = time.time()
        feedbacks = feedbacks.to(device=communication_device)
        reqs_send: List[Future] = []
        reqs_recv: List[Future] = []
        for n in range(N):
            # Get ready to receive feedback from the worker
            logging.info(f"Server {rank} receiving feedback from worker {n+1}")
            req = dist.irecv(tensor=feedbacks[n], src=n+1)
            reqs_recv.append(req)

            # Send the generated data to the worker
            X_g = K[n % k]
            X_d = K[(n + 1) % k] # The results are reflects a version with the following line: X_d = K[n % k + 1]

            # Concatenate the generated data with the feedback
            t_n = torch.stack([X_g, X_d], dim=0)
            t_n = t_n.to(device=communication_device)
            logging.info(f"Server {rank} sending generated data with shape {t_n.shape} to worker {n+1}")
            req = dist.isend(tensor=t_n, dst=n+1)
            reqs_send.append(req)
            current_logs["size.sent"] += t_n.element_size() * t_n.nelement() / 1024**2 # in MB
            logging.info(f"Server {rank} sent data to worker {n}")

        # Wait for all the data to be sent
        for n in range(N):
            reqs_send[n].wait()
        current_logs["end.send_data"] = time.time()
        logging.info(f"Server {rank} sent data to all workers")

        # Wait for all the feedback to be received
        current_logs["start.recv_data"] = time.time()
        for n in range(N):
            reqs_recv[n].wait()
        logging.info(f"Server {rank} received feedback from all workers")
        # Migrate the feedbacks to the device (could be the GPU, the gloo backend enforce to receive on CPU)
        feedbacks = feedbacks.to(device=device)
        current_logs["end.recv_data"] = time.time()
        current_logs["size.recv"] = feedbacks.element_size() * feedbacks.nelement() / 1024**2 # in MB

        current_logs["start.agg_gradients"] = time.time()
        # Precompute some constants
        inverse_batch_size_N = 1.0 / (batch_size * N)

        # Aggregate gradients for each batch and parameter
        grads_sum = [
            torch.zeros_like(p, requires_grad=False, device=device)
            for p in generator.parameters()
        ]

        # Pre-compute gradients for all feedbacks in a batch
        for n in range(N):
            X_g = K[n % k]

            # Compute gradients for the entire batch at once if possible
            # Flatten X_g and feedback to match the batch dimensions if necessary
            batched_X_g = torch.cat([x_i.unsqueeze(0) for x_i in X_g], dim=0)
            batched_feedback = torch.cat([e_i.unsqueeze(0) for e_i in feedbacks[n]], dim=0)

            # Calculate gradients for the whole batch
            batch_grads = torch.autograd.grad(
                outputs=batched_X_g,
                inputs=generator.parameters(),
                grad_outputs=batched_feedback,
                retain_graph=True,
                allow_unused=True
            )

            # Aggregate the gradients
            for j, grad in enumerate(batch_grads):
                if grad is not None:
                    grads_sum[j] += grad

        # Average the gradients and update delta_w
        delta_w = [
            g * inverse_batch_size_N for g in grads_sum
        ]
        current_logs["end.agg_gradients"] = time.time()
        logging.info(f"Server {rank} aggregated the gradients from all workers")

        current_logs["start.calc_gradients"] = time.time()
        # Apply the aggregated gradients to the generator
        optimizer.zero_grad()
        for i, p in enumerate(generator.parameters()):
            if delta_w[i] is not None:
                p.grad = delta_w[i].detach()  # Ensure the grad doesn't carry history
        optimizer.step()
        current_logs["end.calc_gradients"] = time.time()

        if N > 1:
            # Formula obtained from the MD-GAN paper
            if epoch % swap_interval == 0 and epoch > 0:
                current_logs["swap"] = True
                current_logs["start.swap"] = time.time()
                # Create random non-overlapping pairs of workers
                pairs = torch.randperm(N, device=communication_device, dtype=torch.int).view(-1, 2)
                # Send the pairs to the workers
                for pair in pairs:
                    pair = pair + 1

                    req1 = dist.isend(tensor=pair[0], dst=pair[1])
                    req2 = dist.isend(tensor=pair[1], dst=pair[0])
                    current_logs["size.sent"] += 2 * pair.element_size() / 1024**2 # in MB

                    logging.info(f"Server {rank} choosed to swap workers {pair[0]} and {pair[1]}")
                    req1.wait()
                    req2.wait()
                current_logs["end.swap"] = time.time()

        current_logs["end.epoch_calculation"] = time.time()
        if epoch % log_interval == 0 or epoch == epochs - 1:
            fake_images = X.detach()
            logging.info(f"Server {rank} generated {fake_images.shape} images")
            if fake_images.shape[1] < 3:
                fake_images = fake_images.repeat(1, 3, 1, 1)
            # normalize the images from 0 to 255
            fake_images = (fake_images + 1) * 0.5

            grid_fake = make_grid(
                fake_images, nrow=4, value_range=(0, 1), padding=0
            )
            # Convert the grid to a PIL image
            grid_pil = to_pil_image(grid_fake)
            # Create the save directory if it doesn't exist
            Path(image_save_dir).mkdir(parents=True, exist_ok=True)
            grid_path = Path(image_save_dir) / f"generated_epoch_{epoch}.png"
            grid_pil.save(grid_path)

            fake_images = fake_images[:min(n_samples, len(fake_images))].to(device=evaluation_device)

            current_logs["start.is"] = time.time()
            is_score = _compute_inception_score(fake_images, evaluation_device)
            current_logs["end.is"] = time.time()
            current_logs["is"] = is_score.item()

            current_logs["start.fid"] = time.time()
            fid_score = _compute_fid_score(fake_images, real_images, evaluation_device)
            current_logs["end.fid"] = time.time()
            current_logs["fid"] = fid_score.item()

            weights_path.mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(), weights_path / f"generator_{epoch}.pt")

        current_logs["end.epoch"] = time.time()
        csv_writer.writerow(current_logs)

    # Save the generator model
    save_path = Path("weights") / "generator_final.pt"
    torch.save(generator.state_dict(), save_path)
    logging.info(f"Server saved generator model to {save_path}")

    # Distroy the process group
    dist.destroy_process_group()
    logging.info(f"Server finished training")
