#!/usr/bin/env bash

# 1: original_dim
# 2: latent_dim
# 3: hidden_dim
# 4: encoder_path
# 5: path_data

src/python_code/visualize_model.py $1 $2 $3 $4 "asd" $5 "VAE"