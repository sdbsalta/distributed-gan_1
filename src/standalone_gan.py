import argparse
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import time
import random
import numpy as np
from datasets.DataPartitioner import DataPartitioner
import importlib
import csv

def _weights_init(m: nn.Module) -> None:
    """
    Initialize the weights of the network
    :param m: The network
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
    return fid.compute()


def _compute_inception_score(fake_images: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute the inception score
    :param fake_images: The generated images
    :return: The inception score
    """
    inception = InceptionScore(normalize=True, splits=1).to(device)
    inception.update(fake_images)
    return inception.compute()[0]


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None, help="Custom dataset root directory (only used for Custom dataset)")
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--local_epochs", type=int, default=10)
parser.add_argument("--model", type=str, default="cifar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--n_samples_fid", type=int, default=10)
parser.add_argument("--generator_lr", type=float, default=0.0002)
parser.add_argument("--discriminator_lr", type=float, default=0.0002)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--beta_1", type=float, default=0.0)
parser.add_argument("--beta_2", type=float, default=0.999)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.mps.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

device = torch.device(args.device)

if __name__ == "__main__":
    # Determine the evaluation device
    evaluation_device = torch.device("cpu")
    if torch.cuda.is_available():
        evaluation_device = torch.device("cuda")
    
    print(f"Running in {device} and evaluating on {evaluation_device}")

    # Dynamically import the dataset module
    dataset_module = importlib.import_module(f"datasets.{args.dataset}")

    # Initialize the generator and discriminator
    generator: nn.Module = dataset_module.Generator().to(device)
    discriminator: nn.Module = dataset_module.Discriminator().to(device)

    # Initialize the weights of the generator and discriminator
    generator.apply(_weights_init)
    discriminator.apply(_weights_init)

    # Print the summary of the models
    print(discriminator)
    print(generator)

    # Retrieve the image shape and z dimension
    z_dim: int = dataset_module.Z_DIM

    # Retrieve the arguments of the program
    batch_size: int = args.batch_size
    n_samples_fid: int = args.n_samples_fid
    epochs: int = args.epochs
    local_epochs: int = args.local_epochs
    generator_lr: float = args.generator_lr
    discriminator_lr: float = args.discriminator_lr
    beta_1: float = args.beta_1
    beta_2: float = args.beta_2

    # Initialize the data partitioner
    if args.dataset.lower() == "custom":
        partioner = dataset_module.Partitioner(0, 0, path=args.data_dir)
    else:
        partioner = dataset_module.Partitioner(0, 0)
    partioner.load_data()
    dataloader = DataLoader(
        partioner.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    dataloader_it = iter(dataloader)

    # Setup the loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=(beta_1, beta_2)
    )
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(beta_1, beta_2)
    )

   

    image_output_path: Path = Path("saved_images_standalone")
    weights_output_path: Path = Path("weights")
    logs_output_path: Path = Path("logs")
    logs_file = logs_output_path / f"{args.dataset}.standalone.logs.csv"

    get_log = lambda epoch: {
        "epoch": epoch,
        "start.epoch": time.time(),
        "end.epoch": None,
        "start.epoch_calculation": time.time(),
        "start.discriminator_train": None,
        "end.discriminator_train": None,
        "start.generator_train": None,
        "start.generator_train": None,
        "start.generate_data": None,
        "end.generate_data": None,
        "end.generator_train": None,
        "end.epoch_calculation": None,
        "start.calc_gradients": None,
        "end.calc_gradients": None,
        "absolut_step": epoch * local_epochs,
        "mean_d_loss": None,
        "mean_g_loss": None,
        "start.train": time.time(),
        "end.train": None,
        "start.fid": None,
        "end.fid": None,
        "start.is": None,
        "end.is": None,
        "fid": None,
        "is": None,
    }

    f = open(logs_file, "a", encoding="utf-8")
    csv_writer = csv.DictWriter(f, fieldnames=get_log(0).keys())
    csv_writer.writeheader()
    noise_for_saving = torch.randn(batch_size, z_dim, 1, 1, device=device)

    print(f"Entering training loop for {epochs} epochs")
    for epoch in range(epochs):
        current_logs = get_log(epoch)
        current_logs["start.generate_data"] = time.time()

        try:
            real_images = next(dataloader_it)[0].to(device)
        except StopIteration:
            dataloader_it = iter(dataloader)
            real_images = next(dataloader_it)[0].to(device)

        current_logs["end.generate_data"] = time.time()

        losses_d = torch.zeros(local_epochs, device=device, dtype=torch.float32)
        losses_g = torch.zeros(local_epochs, device=device, dtype=torch.float32)

        current_logs["start.calc_gradients"] = time.time()
        for i in range(local_epochs):
            # === Train Discriminator ===
            discriminator.zero_grad()
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)   # fresh noise each step
            fake_images = generator(noise)

            output = discriminator(real_images)
            real_label = torch.ones_like(output, device=device)
            errD_real = criterion(output, real_label)
            errD_real.backward()

            output = discriminator(fake_images.detach())
            fake_label = torch.zeros_like(output, device=device)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            losses_d[i] = errD
            optimizer_discriminator.step()

            # === Train Generator ===
            generator.zero_grad()
            output = discriminator(fake_images)
            real_label = torch.ones_like(output, device=device)
            errG = criterion(output, real_label)
            losses_g[i] = errG
            errG.backward()
            optimizer_generator.step()

            print(f"Epoch {epoch}, Step {i}, Loss D {errD.item():.4f}, Loss G {errG.item():.4f}")

        current_logs["end.calc_gradients"] = time.time()
        current_logs["mean_d_loss"] = losses_d.mean().item()
        current_logs["mean_g_loss"] = losses_g.mean().item()

        # === Save image using fixed noise ===
        with torch.no_grad():
            fake_images_save = generator(noise_for_saving).detach().clone()

        fake_images_save = (fake_images_save + 1) * 0.5
        if fake_images_save.shape[1] < 3:
            fake_images_save = fake_images_save.repeat(1, 3, 1, 1)

        fake_images_save = fake_images_save[:args.n_samples_fid].to(device=evaluation_device)

        image_output_path.mkdir(parents=True, exist_ok=True)
        try:
            grid = make_grid(fake_images_save, nrow=4, normalize=True, value_range=(0, 1), padding=0)
            grid_pil = to_pil_image(grid.cpu())
            grid_pil.save(image_output_path / f"fake_samples_{epoch}.png")
            print(f"[✅] Saved fake_samples_{epoch}.png")
        except Exception as e:
            print(f"[❌] Failed to save image for epoch {epoch}: {e}")

        # === Save logs and weights ===
        logs_output_path.mkdir(parents=True, exist_ok=True)
        current_logs["end.epoch"] = time.time()
        csv_writer.writerow(current_logs)

        weights_output_path.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(), weights_output_path / f"netG_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), weights_output_path / f"netD_epoch_{epoch}.pth")




        print(
            f"Epoch {epoch}, Step {i}, Loss D {errD.item()}, Loss G {errG.item()}"
        )
        print(f"[DEBUG] About to save image for epoch {epoch}...")


        if True:
            # Always save image every epoch
            image_output_path.mkdir(parents=True, exist_ok=True)

            # Normalize images to [0, 1]
            real_images = (real_images + 1) * 0.5
            fake_images = (fake_images + 1) * 0.5

            # Repeat grayscale channels to match FID input
            if fake_images.shape[1] < 3:
                fake_images = fake_images.repeat(1, 3, 1, 1)
            if real_images.shape[1] < 3:
                real_images = real_images.repeat(1, 3, 1, 1)

            # Trim samples for FID & Inception
            real_images = real_images[: args.n_samples_fid].to(device=evaluation_device)
            fake_images = fake_images[: args.n_samples_fid].to(device=evaluation_device)

            # Save sample grid
            try:
                grid = make_grid(fake_images, nrow=4, normalize=True, value_range=(0, 1), padding=0)
                grid_pil = to_pil_image(grid.cpu())
                grid_pil.save(image_output_path / f"fake_samples_{epoch}.png")
                print(f"[✅] Saved fake_samples_{epoch}.png")
            except Exception as e:
                print(f"[❌] Failed to save image for epoch {epoch}: {e}")

        logs_output_path.mkdir(parents=True, exist_ok=True)
        current_logs["end.epoch"] = time.time()
        csv_writer.writerow(current_logs)

    # Check pointing for every epoch
    weights_output_path.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), weights_output_path / f"netG_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), weights_output_path / f"netD_epoch_{epoch}.pth")
