#!/usr/bin/env bash

# 1: original_dim
# 2: latent_dim
# 3: hidden_dim
# 4: encoder_path
# 5: decoder_path
# 6: bijecter_path
# 7: path_data
# 8: scaler_path

src/python_code/visualize_model.py $1 $2 $3 $4 $5 $7 "PAE" $6 $8