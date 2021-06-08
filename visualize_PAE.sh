#!/usr/bin/env bash


original_dim=784
latent_dim=2
hidden_dim=64
path="Outputs/6eYSHL/models/"
encoder_path="${path}model_32_PAE_encoder\$z_2\$h_64.h5"
decoder_path="${path}model_32_PAE_decoder\$z_2\$h_64.h5"
bijecter_path="${path}model_32_PAE_b\$z_2\$h_64.h5"
scaler_path="${path}model_32_PAEscaler.pkl"

save_path="tmp_data/"
#mkdir $save_path
#src/python_code/no_filter.py "${save_data}" "MNIST"
src/do_visualizer_PAE.sh $original_dim $latent_dim $hidden_dim $encoder_path $decoder_path $bijecter_path $save_path $scaler_path

#rm -r $save_path