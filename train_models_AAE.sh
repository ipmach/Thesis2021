#!/usr/bin/env bash


main_folder="Outputs"
set_folder=$(src/python_code/create_set.py "File_manager/")
manager_folder="File_manager/${set_folder}/"

filter_folder="${main_folder}/${set_folder}/FilterData"
models_folder="${main_folder}/${set_folder}/${1:-localrun}Models_AAE"
plot_folder="${main_folder}/${set_folder}/${1:-localrun}Plots_AAE"
logs_folder="${main_folder}/${set_folder}/${1:-localrun}logs_AAE"
value_1=3
value_2=7
N=10

echo "Creating folders for plots and models"
mkdir "${main_folder}/${set_folder}"
mkdir $models_folder
mkdir $plot_folder

echo "Starting tensorboard"
mkdir $logs_folder
mkdir "${logs_folder}/gradient_tape"
#tensorboard --logdir "${1:-localrun}logs_AAE/gradient_tape/"  --host=127.0.0.1 &

echo "Filtering database"
mkdir $filter_folder
mkdir "${filter_folder}/1"
src/python_code/filter.py "${filter_folder}/1" MNIST "FilterValues" $value_1 $value_2

echo "Start training"
for i in $(seq 1 $N)
do
  echo "$i/$N"
   src/train_AAE_2_64.sh 1 $filter_folder $models_folder $plot_folder $logs_folder $manager_folder "AAE"
done

echo "Removing folder database"
rm -r $filter_folder