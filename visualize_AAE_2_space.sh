#!/usr/bin/env bash

value_1=0
value_2=9
original_dim=784
latent_dim=2
hidden_dim=64
path="Outputs/uuyayZ/localrunModels/"
encoder_path="${path}model_AAE_${value_1}_${value_2}_encoder\$z_2\$h_64.h5"
decoder_path="${path}model_AAE_${value_1}_${value_2}_decoder\$z_2\$h_64.h5"
discriminator_path="${path}model_AAE_${value_1}_${value_2}_discriminator\$z_2\$h_64.h5"

save_path="tmp_data/"
mkdir $save_path
#src/python_code/filter_DB_onces.py $value_1 $value_2 "${save_path}"
src/python_code/filter.py "${save_path}" "FashionMnist" "FilterValues" $value_1 $value_2
src/do_visualizer_AAE.sh $original_dim $latent_dim $hidden_dim $encoder_path $decoder_path $discriminator_path $save_path "None"

rm -r $save_path



