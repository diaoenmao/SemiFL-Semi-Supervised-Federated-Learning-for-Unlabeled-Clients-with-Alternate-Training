# SemiFL: Semi-Supervised Federated Learning for Unlabeled Clients with Alternate Training
This is an implementation of [SemiFL: Semi-Supervised Federated Learning for Unlabeled Clients with Alternate Training](https://arxiv.org/abs/2106.01432)
- A resourceful server with labeled data can significantly improve its learning performance by working with distributed clients with unlabeled data without data sharing.
<p align="center">
<img src="/asset/SSFL.png">
</p>
- An illustration of (a) vanilla combination of communication efficient FL and SSL, and (b) Alternate Training (Ours).
<p align="center">
<img src="/asset/SemiFL.png">
</p>

## Requirements
See requirements.txt

## Instructions
 - Global hyperparameters are configured in `config.yml`
 - Use `make.sh` to generate run script
 - Use `make.py` to generate exp script
 - Use `process.py` to process exp results
 - Experimental setup are listed in `make.py` 
 - Hyperparameters can be found at `process_control()` in utils.py 
 - `modules/modules.py` defines Server and Client
    - sBN statistics are updated in `distribute()` of Server
    - global momemtum is used in `update()` of Server
    - fix and mix dataset are constructed in `make_dataset()` of Client
 - The data are split at `split_dataset()` in `data.py`
 
## Examples
 - Train SemiFL for CIFAR10 dataset (WResNet28x2, $N_\mathcal{S}=4000$, fix ( $\tau=0.95$ ) and mix loss, $M=100$, $C=0.1$, IID, $E=5$, server and client sBN statistics)
    ```ruby
    python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 4000_fix@0.95-mix_100_0.1_iid_5_0.5_1
    ```
 - Train SemiFL for CIFAR10 dataset (WResNet28x2, $N_\mathcal{S}=250$, fix ( $\tau=0.95$ ) and mix loss, $M=100$, $C=0.1$, Non-IID ( $K=2$ ), $E=5$, server and client sBN statistics)
    ```ruby
    python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 250_fix@0.95-mix_100_0.1_non-iid-l-2_5_0.1_1
    ```
 - Test SemiFL for SVHN dataset (WResNet28x2, $N_\mathcal{S}=1000$, fix ( $\tau=0.95$ ) loss, $M=100$, $C=0.1$, Non-IID ( $Dir(0.3)$ ), $E=5$, server only sBN statistics)
    ```ruby
    python test_classifier_ssfl.py --data_name SVHN --model_name wresnet28x2 --control_name 1000_fix@0.95_100_0.1_non-iid-d-0.3_1_0_0
    ```
    
## Results
- Results of CIFAR10 dataset with (a) $N_{\mathcal{S}} = 250$ and (b) $N_{\mathcal{S}} = 4000$.
<p align="center">
<img src="/asset/CIFAR10.png">
</p>

## Acknowledgements
*Enmao Diao  
Jie Ding  
Vahid Tarokh*