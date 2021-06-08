#!/usr/bin/env bash

main_folder="Outputs"
set_folder=$(src/python_code/create_set.py "File_manager/")
manager_folder="File_manager/${set_folder}/"
save_data="${main_folder}/${set_folder}/save_data"
save_model="${main_folder}/${set_folder}/VAE_models"
logs_VAE="${main_folder}/${set_folder}/logs_VAE"
plot_VAE="${main_folder}/${set_folder}/plot_VAE"

mkdir "${main_folder}/${set_folder}"
mkdir $save_data
mkdir $save_model
mkdir $logs_VAE
mkdir $plot_VAE


mkdir "${logs_VAE}/gradient_tape"
#tensorboard --logdir "${logs_VAE}/train/gradient_tape/"  --host=127.0.0.1 &

mkdir "${save_data}/all"
src/python_code/no_filter.py "${save_data}/all" "MNIST"
src/train_VAE_2_64.sh "all" $save_data $save_model $plot_VAE $logs_VAE $manager_folder "all"
for num in {0..9}
do
  mkdir "$save_data/${num}"
  src/python_code/filter.py "$save_data/${num}" "MNIST" "FilterValues" $num

  src/train_VAE_2_64.sh $num $save_data $save_model $plot_VAE $logs_VAE $manager_folder $num
  #src/train_VAE_4_64.sh $num $save_data $save_model $plot_VAE $logs_VAE $manager_folder
done
rm -r $save_data