#!/bin/bash
. ./shared-args.sh
cd src

seed=1

python standalone_gan.py --local_epochs $local_epochs \
    --epochs $epochs \
    --model $model \
    --dataset $dataset \
    --generator_lr $generator_lr \
    --discriminator_lr $discriminator_lr \
    --device $device \
    --batch_size $batch_size \
    --seed $seed \
    --log_interval $log_interval
