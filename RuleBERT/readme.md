

# RuleBERT Training Repository

Welcome to the RuleBERT training repository. This README provides a guide on how to set up and use the repository for training the RuleBERT models.

## Installation

1. Clone this repository to your local machine using:
   
```
!git clone https://github.com/HLR/Probabilistic_Logical_Reasoning.git
```

2. Ensure you have the requirements in the [requirements.txt](https://github.com/HLR/Probabilistic_Logical_Reasoning/blob/main/requirements.txt) installed.
   
```
pip install -r requirements.txt
```

## Dataset 

Run the bash file download_dataset.sh to download the dataset or visit [RuleBERT Dataset Repository](https://github.com/MhmdSaiid/RuleBert).

## Training

The script `train.py` facilitates the training of RuleBERT models, incorporating techniques like PCT.

For a basic RuleBERT training session Similar to RuleBERTs original paper use these paramteres:

```
!cd RuleBERT
!python train_rulebert.py --lr 1e-6 --use_ruletext --losstype wBCE
```

For a baseline RuleBERT training session Similar to our method use these paramteres:

```
!cd RuleBERT
!python train_rulebert.py --lr 1e-5 --use_ruletext --losstype wBCE --freeze_layers 
```

To include PCT use these paramteres:

```
!cd RuleBERT
!python train_rulebert.py --lr 1e-5 --use_ruletext --losstype wBCE --freeze_layers --apply_PCT --alpha 0.01
```

The script supports a range of customizable command line arguments to refine the training process:

- `--seed`: Initialize the random seed for reproducibility.
- `--include_first`: Include the initial rule in PCT for foundational training.
- `--chain_number`: Specify the depth of reasoning chains.
- `--cuda_number`: Assign the CUDA device for GPU acceleration.
- `--apply_PCT`: Activate PCT.
- `--use_ruletext`: Incorporate rule texts directly into the training.
- `--losstype`: Choose the loss function for model optimization.
- `--model_name_saved`: Name the model upon saving.
- `--data_dir`: Designate the directory for training and validation data.
- `--lr`: Set the learning rate for the optimizer.
- `--alpha`: Adjust the PCT hyperparameter for balance in training.
- `--freeze_layers`: Freeze certain layers for efficiency in training.

### Utility Support

Accompanying the training script are utility modules (`utils.py`, `training_utils.py`).


