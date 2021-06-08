#!/usr/bin/env bash

num_epochs=80
batch_size=100
learning_rate="0.001"
train_buf=60000
dropout="0.5"
model_type="AAE"

src/python_code/train_script.py $1 $2 $3 $4 $5 $6 ${10} $8 $9 $num_epochs $batch_size $learning_rate $7 $model_type $train_buf $dropout