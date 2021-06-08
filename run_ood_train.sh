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

hidden_dim=(64 128 256 512 1024)
latent_space=$1
image_size=3072

echo 'Starting run with latent space ', $latent_space

mkdir "${main_folder}/${set_folder}"
mkdir $save_data
mkdir $save_model
mkdir $logs_VAE
mkdir $plot_VAE

mkdir "${logs_VAE}/gradient_tape"
#tensorboard --logdir "${logs_VAE}/train/gradient_tape/"  --host=127.0.0.1 &

mkdir "${save_data}/all"
#src/python_code/no_filter.py "${save_data}/all" "MNIST" "MNIST-C" "FashionMnist"
src/python_code/no_filter.py "${save_data}/all" "cifar10" "cifar10-C" "svhn_cropped"

for h in ${hidden_dim[@]}; do
    echo $latent_space, $h
    src/train_Vanilla.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "Vanilla" $h $latent_space $image_size
    src/train_SAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "SAE" $h  $latent_space $image_size
    src/train_DAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "DAE" $h  $latent_space $image_size
    src/train_VAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "VAE" $h  $latent_space $image_size
    src/train_AAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "AAE" $h  $latent_space $image_size
    src/train_PAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "PAE" $h  $latent_space $image_size
    src/train_EAE.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "EAE" $h  $latent_space $image_size
done


rm -r "${save_data}"