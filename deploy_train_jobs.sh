#!/usr/bin/env bash

docker build --tag interpolation_train:snapshot .

#for run in {0..4}
#do

#TODO Revert
run=0

for process in {1..5}
do
  RUNID=$((process+(run*5)))
  #mkdir /experiments/andre/ood_aes/${RUNID}/
  #docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train.sh /res/${RUNID}/
  docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/train_models.sh /res/${RUNID}/
  #docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/train_models_AAE.sh /res/${RUNID}/
done

for process in {1..5}
do
  RUNID=$((process+(run*5)))
  echo "Waiting for ipt_${RUNID} to finish"
  docker wait ipt_${RUNID}
  docker rm ipt_${RUNID}
done

for process in {1..5}
do
  RUNID=$((process+(run*5)))
  #mkdir /experiments/andre/ood_aes/${RUNID}/
  #docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/run_ood_train.sh /res/${RUNID}/
  docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/train_models.sh /res/${RUNID}/
  #docker run -dit --name ipt_${RUNID} --gpus all -v /experiments/andre/interpolation_training:/res interpolation_train:snapshot /code/train_models_AAE.sh /res/${RUNID}/
done

for process in {1..5}
do
  RUNID=$((process+(run*5)))
  echo "Waiting for ipt_${RUNID} to finish"
  docker wait ipt_${RUNID}
  docker rm ipt_${RUNID}
done
##done
