#!/usr/bin/env bash

# 1: original_dim
# 2: latent_dim
# 3: hidden_dim
# 4: encoder_path
# 5: decoder_path
# 6: discriminator_path
# 7: path_data
# 8: path_tree or None

src/python_code/visualize_model.py $1 $2 $3 $4 $5 $7 "AAE" $6 $8