#!/bin/bash
#SBATCH -D /users/adck705/IN3062-Introduction-to-AI
#SBATCH --job-name my-gputest
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

#Modules required
#This is an example you need to select the modules your code needs.
module load python/3.7.12
module load libs/nvidia-cuda/11.2.0/bin

#Run your script.
python3 coursework.py