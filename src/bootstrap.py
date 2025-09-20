import torch
import logging
import argparse
import os
from typing import Tuple, Dict, Any
from pathlib import Path
import torch.nn as nn
import random
import numpy as np
import importlib
from datasets.DataPartitioner import DataPartitioner
from actors import worker, server
import torch.multiprocessing as mp
from typing import List, Callable


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


parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="nccl")
parser.add_argument("--world_size", type=int, default=2)
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--ranks", type=str, default="0,1,2")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--swap_interval", type=int, default=1)
parser.add_argument("--local_epochs", type=int, default=10)
parser.add_argument("--model", type=str, default="cifar")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--generator_lr", type=float, default=0.001)
parser.add_argument("--discriminator_lr", type=float, default=0.004)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--master_addr", type=str, default="localhost")
parser.add_argument("--master_port", type=str, default="1234")
parser.add_argument("--network_interface", type=str, required=False)
parser.add_argument("--iid", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--beta_1", type=float, default=0.0)
parser.add_argument("--beta_2", type=float, default=0.999)
args = parser.parse_args()

log_folder: Path = Path("logs")
log_folder.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

os.environ["MASTER_ADDR"] = args.master_addr
os.environ["MASTER_PORT"] = args.master_port
os.environ["WORLD_SIZE"] = str(args.world_size)
if args.network_interface:
    os.environ["GLOO_SOCKET_IFNAME"] = args.network_interface
    os.environ["NCCL_SOCKET_IFNAME"] = args.network_interface
os.environ["USE_CUDA"] = "1"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["GLOO_LOG_LEVEL"] = "DEBUG"
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def run(rank: int, args: argparse.Namespace, partioner: DataPartitioner, image_shape: Tuple[int, int, int], z_dim: int, generator: nn.Module, discriminator: nn.Module) -> None:
    # If the rank is greater than 0, we are a worker
    logging.info(f"Starting up process for node: {rank}")
    if rank > 0:
        # Initialize dataset with world size-1 because the server should not count as a worker
        discriminator = discriminator().to(device=args.device, dtype=torch.float32)
        discriminator.apply(_weights_init)

        worker.start(
            backend=args.backend,
            rank=rank,
            world_size=args.world_size,
            batch_size=args.batch_size,
            swap_interval=args.swap_interval,
            data_partitioner=partioner,
            epochs=args.epochs,
            discriminator=discriminator,
            device=args.device,
            local_epochs=args.local_epochs,
            image_shape=image_shape,
            log_interval=args.log_interval,
            generator=None,
            discriminator_lr=args.discriminator_lr,
            generator_lr=args.generator_lr,
            z_dim=z_dim,
            log_folder=log_folder,
            dataset_name=args.dataset,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
        )
    else:
        # If the rank is 0, we are the server
        generator = generator().to(device=args.device, dtype=torch.float32)
        generator.apply(_weights_init)

        server.start(
            backend=args.backend,
            rank=rank,
            world_size=args.world_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            generator=generator,
            dataset=partioner.train_dataset,
            device=args.device,
            image_shape=image_shape,
            generator_lr=args.generator_lr,
            z_dim=z_dim,
            log_interval=args.log_interval,
            log_folder=log_folder,
            iid=args.iid == 1,
            dataset_name=args.dataset,
            swap_interval=args.swap_interval,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
        )

def init_process(local_rank: int, args: argparse.Namespace, ranks: List[int], partioner: DataPartitioner, image_shape: Tuple[int, int, int], z_dim: int, generator: nn.Module, discriminator: nn.Module, fn: Callable[[int, argparse.Namespace], None]) -> None:
    """
    Initialize the process
    :param rank: The rank of the process
    :param world_size: The world size
    :param args: The arguments of the program
    """
    rank = ranks[local_rank]
    os.environ["RANK"] = str(rank)

    partioner.rank = rank

    seed_shifted = args.seed + rank
    np.random.seed(seed_shifted)
    random.seed(seed_shifted)
    torch.manual_seed(seed_shifted)
    torch.mps.manual_seed(seed_shifted)
    torch.cuda.manual_seed(seed_shifted)
    torch.cuda.manual_seed_all(seed_shifted)
    torch.backends.cudnn.deterministic = True

    fn(rank, args, partioner, image_shape, z_dim, generator, discriminator)

if __name__ == "__main__":
    ranks: List[int] = []
    if ".." in args.ranks:
        start, end = args.ranks.split("..")
        ranks = list(range(int(start), int(end) + 1))
    elif "," in args.ranks:
        ranks = [int(rank) for rank in args.ranks.split(",")]
    elif args.ranks.isdigit():
        ranks = [int(args.ranks)]
    else:
        raise ValueError("Invalid rank format")

    logging.info(f"{len(ranks)} ranks: {', '.join([str(rank) for rank in ranks])}")

    if args.world_size % 2 == 0:
        raise ValueError("World size must be odd")

    # Dynamically import the dataset module
    dataset_module = importlib.import_module(f"datasets.{args.dataset}")

    # Initialize the data partitioner
    partioner: DataPartitioner = dataset_module.Partitioner
    partioner = partioner(args.world_size, 0)
    partioner.load_data()

    # Initialize the generator and discriminator
    generator: nn.Module = dataset_module.Generator
    discriminator: nn.Module = dataset_module.Discriminator

    # Retrieve the image shape and z dimension
    image_shape: Tuple[int, int, int] = dataset_module.SHAPE
    z_dim: int = dataset_module.Z_DIM
    
    mp.spawn(
        init_process,
        args=(args, ranks, partioner, image_shape, z_dim, generator, discriminator, run),
        nprocs=len(ranks),
        join=True,
    )
