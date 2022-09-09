#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name SVHN --model_name wresnet28x2 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1
wait
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 1 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 2 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_iid_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.3_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-d-0.1_5-0_0.5_1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --data_name CIFAR100 --model_name wresnet28x8 --init_seed 3 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name fs_sup_100_0.1_non-iid-l-2_5-0_0.5_1
wait
