#!/usr/bin/env bash

main_folder="/res/Outputs"
set_folder=$(src/python_code/create_set.py "/res/File_manager/")
manager_folder="/res/File_manager/${set_folder}/"

permutation_file="Permutation.npy"
filter_folder="${main_folder}/${set_folder}/FilterData"
models_folder="${main_folder}/${set_folder}/Models"
plot_folder="${main_folder}/${set_folder}/Plots"
logs_folder="${main_folder}/${set_folder}/logs"
output_folder="${main_folder}/${set_folder}/outputs"
size=10
latent_dimension=2
hidden_dimension=128
# Only AAE
experiment_folder="real_distribution.json"

echo "Creating folders for plots and models"
mkdir "${main_folder}/${set_folder}"
mkdir $models_folder
mkdir $plot_folder
mkdir $output_folder
mkdir $filter_folder

echo "Starting tensorboard"
mkdir $logs_folder
mkdir "${logs_folder}/gradient_tape"
#tensorboard --logdir "${logs_folder}/train/gradient_tape/"  --host=127.0.0.1 &

echo "Filtering database"
mkdir $filter_folder
# Creating subfolders
N=$(($size-2))
for i in $(seq 0 $N)
do
  mkdir "$filter_folder/$i"
done

echo "Start training"
#src/python_code/permutation.py $size $N $permutation_file
#src/python_code/filter.py $filter_folder "FashionMnist" "Permutation" $permutation_file 0
#src/train_AAE_2_64.sh 0 $filter_folder $models_folder $plot_folder $logs_folder $manager_folder "AAE"


for num in {0..9}
do
  src/python_code/permutation.py $size $num $permutation_file
  src/python_code/filter.py $filter_folder "MNIST" "Permutation" $permutation_file $num
  for i in $(seq 0 $N)
  do
    echo "$i/$N"
    src/train_AAE_2_64.sh $i $filter_folder $models_folder $plot_folder $logs_folder $manager_folder "AAE"
  done
  N=$((N-1))
done

# Generate output images
src/python_code/main_cmp.py $models_folder $output_folder $latent_dimension $hidden_dimension

echo "Removing folder database"
rm -r $filter_folder
rm $permutation_file

