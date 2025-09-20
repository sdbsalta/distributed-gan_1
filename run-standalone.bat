@echo off
REM === Set custom arguments ===
setlocal

set LOCAL_EPOCHS=1
set EPOCHS=500
set MODEL=DCGAN
set DATASET=Custom
set DATA_DIR=C:\Users\Rafaela\Desktop\Thesis\finds\00000
set GENERATOR_LR=0.0002
set DISCRIMINATOR_LR=0.0002
set DEVICE=cuda
set BATCH_SIZE=16
set LOG_INTERVAL=10
set SEED=1

REM === Move into src directory ===
cd src

python standalone_gan.py ^
 --local_epochs %LOCAL_EPOCHS% ^
 --epochs %EPOCHS% ^
 --model %MODEL% ^
 --dataset %DATASET% ^
 --data_dir "%DATA_DIR%" ^
 --generator_lr %GENERATOR_LR% ^
 --discriminator_lr %DISCRIMINATOR_LR% ^
 --device %DEVICE% ^
 --batch_size %BATCH_SIZE% ^
 --seed %SEED% ^
 --log_interval %LOG_INTERVAL%

endlocal
