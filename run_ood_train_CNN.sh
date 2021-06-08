#!/usr/bin/env bash

mkdir "/res/Outputs"
mkdir "/res/File_manager"

main_folder="/res/Outputs"
set_folder=$(src/python_code/create_set.py "/res/File_manager/")
manager_folder="/res/File_manager/${set_folder}/"
save_data="${main_folder}/${set_folder}/save_data"
save_model="${main_folder}/${set_folder}/models"
logs_VAE="${main_folder}/${set_folder}/logs"
plot_VAE="${main_folder}/${set_folder}/plot"

hidden_dim=128
latent_space=$1
image_size=3072

mkdir "${main_folder}/${set_folder}"
mkdir $save_data
mkdir $save_model
mkdir $logs_VAE
mkdir $plot_VAE

mkdir "${logs_VAE}/gradient_tape"
#tensorboard --logdir "${logs_VAE}/train/gradient_tape/"  --host=127.0.0.1 &

mkdir "${save_data}/all"
src/python_code/no_filter.py "${save_data}/all" "MNIST" "MNIST-C" "FashionMnist"
#src/python_code/no_filter.py "${save_data}/all" "cifar10" "cifar10-C" "svhn_cropped"

src/train_Vanilla_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "Vanilla_CNN" $hidden_dim $latent_space $image_size
src/train_SAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "SAE_CNN" $hidden_dim  $latent_space $image_size
src/train_DAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "DAE_CNN" $hidden_dim  $latent_space $image_size
src/train_VAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "VAE_CNN" $hidden_dim  $latent_space $image_size
src/train_AAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "AAE_CNN" $hidden_dim  $latent_space $image_size
src/train_PAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "PAE_CNN" $hidden_dim  $latent_space $image_size
src/train_EAE_CNN.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "EAE_CNN" $hidden_dim  $latent_space $image_size

rm -r "${save_data}"