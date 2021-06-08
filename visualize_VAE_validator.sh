#!/usr/bin/env bash

value_1=4
original_dim=784
latent_dim=2
hidden_dim=64
path="Outputs/pqWqdE/VAE_models/"

model="${path}model_32_[${value_1}]_[${value_1}].h5"

save_path="tmp_data/"
mkdir $save_path
src/python_code/filter_DB_one.py $value_1 "${save_path}"
src/do_visualizer_VAE.sh $original_dim $latent_dim $hidden_dim $model $save_path

rm -r $save_path