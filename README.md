# SemiFL: Communication Efficient Semi-SupervisedFederated Learning with Unlabeled Clients
This is an implementation of SemiFL: Communication Efficient Semi-SupervisedFederated Learning with Unlabeled Clients
 
## Requirements
See requirements.txt

## Instruction
 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 
 - modules/modules.py defines Server and Client
    - sBN statistics are updated in distribute() of Server
    - global momemtum is used in update() of Server
    - fix and mix dataset are constructed in make_dataset() of Client
 - The data are split at split_dataset() in data.py

## Examples
 - Train SemiFL for CIFAR10 dataset (WResNet28x2, $$N_\mathcal{S}=4000$$, fix and mix loss, $$M=100$$, $$C=0.1$$, IID, $$E=5$$, $$\beta_g=0.5$$, server and client sBN statistics)
    ```ruby
    python train_model_fed.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 4000_fix-mix_100_0.1_iid_5_0.5_1
    ```
 - Train SemiFL for CIFAR10 dataset (WResNet28x2, $$N_\mathcal{S}=250$$, fix and mix loss, $$M=100$$, $$C=0.1$$, Non-IID ($$K=2$$), $$E=5$$, $$\beta_g=0.1$$, server and client sBN statistics)
    ```ruby
    python train_model_fed.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 250_fix-mix_100_0.1_non-iid-l-2_5_0.1_1
    ```
 - Test SemiFL for SVHN dataset (WResNet28x2, $$N_\mathcal{S}=1000$$, fix loss, $$M=100$$, $$C=0.1$$, Non-IID ($$\operatorname{Dir}(0.3)$$), $$E=1$$, $$\beta_g=0$$, server only sBN statistics)
    ```ruby
    python test_model_fed.py --data_name SVHN --model_name wresnet28x2 --control_name 1000_fix_100_0.1_non-iid-d-0.3_1_0_0
    ```