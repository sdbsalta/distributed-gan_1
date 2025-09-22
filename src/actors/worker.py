import torch.distributed as dist
import torch
import logging
import torch.nn as nn
import torch.utils.data
from tensordict import TensorDict
from pathlib import Path
from md_datasets.DataPartitioner import DataPartitioner
from typing import List, Dict, Any, Tuple
import os
import time
from pathlib import Path
import os
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from datetime import timedelta
import csv


def start(
    backend: str,
    rank: int,
    world_size: int,
    data_partitioner: DataPartitioner,
    discriminator_lr: float,
    generator_lr: float,
    epochs: int,
    swap_interval: int,
    local_epochs: int,
    log_interval: int,
    discriminator: torch.nn.Module,
    generator: torch.nn.Module,
    batch_size: int,
    image_shape: Tuple[int, int, int],
    log_folder: Path,
    dataset_name: str,
    device: torch.device = torch.device("cpu"),
    z_dim: int = 100,
    beta_1: float = 0.0,
    beta_2: float = 0.999,
) -> None:
    # Initialize the worker TCP connection
    print(
        f"Connecting worker {rank} to server {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}"
    )
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(weeks=52),
    )
    dist.barrier()

    # Determine the communication device
    communication_device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        communication_device = torch.device("cuda")

    logging.info(
        f"Worker {rank} initialized on device {device}, communication device: {communication_device}"
    )
    print(discriminator)

    name = f"mdgan.{world_size-1}.{dataset_name}"
    logs_file = log_folder / f"{name}.worker.{rank}.logs.csv"

    # Get the indices of the dataset that the server wants the worker to train on
    # 1. Receive the size of the indices tensor to initialize the tensor which will store the indices
    # 2. Receive the indices tensor and store it in the indices variable
    indices_size = torch.zeros(1, dtype=torch.int, device=communication_device)
    dist.recv(tensor=indices_size, src=0)
    logging.info(f"Worker will store {indices_size.item()} entries")
    indices = torch.arange(indices_size.item(), device=communication_device)
    logging.info(f"Worker {rank} waiting for indices with shape {indices.shape}")
    dist.recv(tensor=indices, src=0)

    # Get the subset of the dataset based on the indices the server sent
    partition_train = data_partitioner.get_subset_from_indices(indices, train=True)

    # Create the dataloader to load the real images based on the indices the server sent
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(
        partition_train,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    dataloader_it = iter(dataloader)

    logging.info(
        f"Worker {rank} with length {len(partition_train)} out of {len(data_partitioner.train_dataset)} ({indices})"
    )

    # Initialize the generator loss function and optimizer
    criterion = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=(beta_1, beta_2)
    )

    # Get the ranks of the other workers
    other_workers_rank = list(range(1, world_size))
    other_workers_rank.remove(rank)

    # Start the swap listener threads, one thread for every other worker
    # swap_status = {"rank": -1, "state_dict": None, "stop": False}
    # threads: List[Thread] = []
    # for other_worker in other_workers_rank:
    #     t = Thread(target=_swap_event, args=(discriminator, other_worker, swap_status))
    #     t.start()
    #     threads.append(t)

    # Initialize the labels values for the real and fake images
    real_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)

    # Determine the model size
    # https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in discriminator.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in discriminator.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / 1024**2
    logging.info(f"Worker {rank} model size: {model_size_mb} MB")

    # Initialize the logs
    get_log = lambda epoch: {
        "epoch": epoch,
        "start.epoch": time.time(),
        "end.epoch": None,
        "start.calc_gradients": None,
        "end.calc_gradients": None,
        "start.recv_data": None,
        "end.recv_data": None,
        "start.send": None,
        "end.send": None,
        "start.swap_recv_instruction": None,
        "end.swap_recv_instruction": None,
        "start.load_state_dict": None,
        "end.load_state_dict": None,
        "start.swap_recv": None,
        "end.swap_recv": None,
        "start.swap_send": None,
        "end.swap_send": None,
        "swap_with": None,
        "mean_d_loss": None,
        "size.model": model_size_mb,
        "size.sent": 0,
        "size.recv": 0,
    }
    f = open(logs_file, "a", encoding="utf-8")
    csv_writer = csv.DictWriter(f, fieldnames=get_log(0).keys())
    csv_writer.writeheader()

    for epoch in range(epochs):
        logging.info(f"Worker {rank} starting epoch {epoch}")
        current_logs = get_log(epoch)

        # Get N random samples from the dataset
        try:
            real_images = next(dataloader_it)[0].to(device)  # Ensure real images are on the correct device
        except StopIteration:
            # Reinitialize the iterator if we run out of data
            dataloader_it = iter(dataloader)
            real_images = next(dataloader_it)[0].to(device)

        # Save real images, commented because it use compute resources and is not necessary
        # grid = make_grid(
        #     real_images, nrow=4, normalize=True, value_range=(-1, 1), padding=0
        # )
        # grid_pil = to_pil_image(grid.cpu())
        # grid_path = Path(f"saved_images_worker_{rank}")
        # grid_path.mkdir(parents=True, exist_ok=True)
        # grid_path = grid_path / f"real_epoch_{epoch}.png"
        # grid_pil.save(grid_path)

        # Receive fake images from the server
        current_logs["start.recv_data"] = time.time()
        X_gen = torch.zeros((2, batch_size, *image_shape), dtype=torch.float32)
        dist.recv(tensor=X_gen, src=0)
        X_gen = X_gen.to(device)
        X_g: torch.Tensor = X_gen[0]
        X_d: torch.Tensor = X_gen[1]
        X_g.requires_grad = True
        X_d.requires_grad = True
        logging.info(f"Worker {rank} received data of shape {X_gen.shape}")
        current_logs["end.recv_data"] = time.time()
        current_logs["size.recv"] += X_gen.nelement() * X_gen.element_size() / 1024**2

        current_logs["start.calc_gradients"] = time.time()
        discriminator.train()
        losses = torch.zeros(local_epochs, dtype=torch.float32, device=device)
        for l in range(local_epochs):
            # Train Discriminator with real images
            discriminator.zero_grad()
            output: torch.Tensor = discriminator(real_images)
            d_loss_real: torch.Tensor = criterion(output, real_labels)

            # Train Discriminator with fake images
            output = discriminator(X_d)
            d_loss_fake: torch.Tensor = criterion(output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_discriminator.step()

            # Save the loss
            losses[l] = d_loss

            logging.info(
                f"Worker {rank} finished local iteration {l}, discriminator loss {d_loss_real + d_loss_fake}"
            )
        # Save the mean loss in the logs
        current_logs["mean_d_loss"] = losses.mean().item()
        current_logs["end.calc_gradients"] = time.time()

        current_logs["start.send"] = time.time()
        # Compute output of the discriminator for a given input
        d_loss_eval: torch.Tensor = discriminator(X_g)
        # Compute loss for fake data
        loss_gen: torch.Tensor = criterion(
            d_loss_eval,
            real_labels,
        )
        # Compute gradients
        loss_gen.backward()
        logging.info(
            f"Worker {rank} sending gradients to server with shape {X_g.grad.shape}"
        )
        # Send the gradients to the server
        grad = X_g.grad.to(device=communication_device)
        dist.send(tensor=grad, dst=0)
        current_logs["end.send"] = time.time()
        current_logs["end.epoch"] = time.time()
        current_logs["size.sent"] += grad.nelement() * grad.element_size() / 1024**2

        # Swap state_dict with another worker
        if len(other_workers_rank) > 0:
            if epoch % swap_interval == 0 and epoch > 0:
                # Receive the pair from the server
                current_logs["start.swap_recv_instruction"] = time.time()
                swap_with = torch.zeros(1, dtype=torch.int, device=communication_device)
                dist.recv(tensor=swap_with, src=0)
                current_logs["size.recv"] += swap_with.nelement() * swap_with.element_size() / 1024**2
                swap_with = swap_with.item()
                current_logs["end.swap_recv_instruction"] = time.time()
                current_logs["swap_with"] = swap_with

                logging.info(f"Worker {rank} received swap_with {swap_with}")

                # Receive the state_dict from the other worker
                state_dict = discriminator.to(communication_device).state_dict()
                recv_state_dict: TensorDict = TensorDict(
                    state_dict, batch_size=[]
                ).unflatten_keys(".")
                reqs_recv = recv_state_dict.irecv(src=swap_with, return_premature=True)
                logging.info(f"Worker {rank} waiting for state_dict from worker {swap_with}")

                # Send the state_dict to the other worker
                current_logs["start.swap_send"] = time.time()
                state_dict = discriminator.to(communication_device).state_dict()
                current_state_dict: TensorDict = TensorDict(
                    state_dict, batch_size=[]
                ).unflatten_keys(".")
                current_state_dict.send(dst=swap_with)
                current_logs["end.swap_send"] = time.time()
                current_logs["size.sent"] += model_size_mb
                logging.info(f"Worker {rank} sent state_dict to worker {swap_with}")

                # Wait for the state_dict to be sent
                current_logs["start.swap_recv"] = time.time()
                for req in reqs_recv:
                    req.wait()
                current_logs["end.swap_recv"] = time.time()
                current_logs["size.recv"] += model_size_mb
                logging.info(f"Worker {rank} received state_dict from worker {swap_with}")
                
                # Load the received state_dict
                current_logs["start.load_state_dict"] = time.time()
                discriminator.load_state_dict(recv_state_dict.flatten_keys("."))
                discriminator.to(device)
                current_logs["end.load_state_dict"] = time.time()
                logging.info(f"Worker {rank} loaded state_dict from worker {swap_with}")

        csv_writer.writerow(current_logs)

    # Save the model
    model_path = Path(f"weights/worker_{rank}")
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = model_path / "discriminator.pth"
    torch.save(discriminator.state_dict(), model_path)
    logging.info(f"Worker {rank} saved model to {model_path}")

    # Discard the process group
    dist.destroy_process_group()

    logging.info(f"Worker {rank} finished training")
