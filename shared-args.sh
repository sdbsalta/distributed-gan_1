#!/bin/bash

batch_size=10
discriminator_lr=0.0002
generator_lr=0.0002
dataset=CIFAR10
model=$dataset
epochs=30000
local_epochs=1
iid=1
n_samples_fid=10
device=cpu
log_interval=300
beta_1=0.5
beta_2=0.999
