#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.99-mix_100_0.1_iid_5-5_0.5_1
wait
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 250_fix@0.99-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.5-mix_100_0.1_iid_5-5_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name 4000_fix@0.99-mix_100_0.1_iid_5-5_0.5_1
wait
