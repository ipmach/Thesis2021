#!/usr/bin/env bash

docker build --tag interpolation_train:snapshot .

latent_space=(16 32 56 64 72 128 4 8)

# First batch
gpu_0=${latent_space[0]}
echo "Executing ipt_${gpu_0} in gpu-0"
latent_space=("${latent_space[@]:1}")
docker run -dit --name ipt_${gpu_0} --gpus 0 -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train_CNN.sh $gpu_0
gpu_1=${latent_space[0]}
echo "Executing ipt_${gpu_1} in gpu-1"
latent_space=("${latent_space[@]:1}")
docker run -dit --name ipt_${gpu_1} --gpus 1 -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train_CNN.sh $gpu_1

# Len queue
exec_queue=${#latent_space[@]}
exec_queue=$((exec_queue - 1))

# run_ood_train_CNN.sh: for CNN models, run_ood_train.sh: for perceptrons models

# Rest batch
while ((exec_queue >= 0))
do
  if [ "$( docker container inspect -f '{{.State.Status}}' ipt_${gpu_0} )" != "running" ]; then
    docker rm ipt_${gpu_0}
    gpu_0=${latent_space[$exec_queue]}
    echo "Executing ipt_${gpu_0} in gpu-0"
    docker run -dit --name ipt_${gpu_0} --gpus 0 -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train_CNN.sh $gpu_0
    exec_queue=$((exec_queue - 1))
  fi

  if [ "$( docker container inspect -f '{{.State.Status}}' ipt_${gpu_1} )" != "running" ]; then
    docker rm ipt_${gpu_1}
    gpu_1=${latent_space[$exec_queue]}
    echo "Executing ipt_${gpu_1} in gpu-1"
    docker run -dit --name ipt_${gpu_1} --gpus 1 -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train_CNN.sh $gpu_1
    exec_queue=$((exec_queue - 1))
  fi
  sleep 5
done

docker wait ipt_${gpu_0}
docker rm ipt_${gpu_0}
docker wait ipt_${gpu_1}
docker rm ipt_${gpu_1}

echo "All training executions are finish"

echo "Gathering data"
  docker run -dit --name ipt_gather --gpus 0 -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/src/gather_ood.sh /res/Outputs/ /res/results_obtain.csv
  docker wait ipt_gather
  docker rm ipt_gather
echo "Experiment complete"
