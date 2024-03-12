
# RuleTaker-pro Training Repository

Welcome to the RuleTaker-pro training repository. This README provides a guide on how to set up and use the repository for training the RuleTaker-pro models.

## Installation

1. Clone this repository to your local machine using:
   
```
!git clone https://github.com/HLR/Probabilistic_Logical_Reasoning.git
```

2. Ensure you have the requirements in the [requirements.txt](https://github.com/HLR/Probabilistic_Logical_Reasoning/blob/main/requirements.txt) installed.
   
```
pip install -r requirements.txt
```

## Datasets and Base models

In order to train RuleTaker-pro we need the following models and datasets:
- RuleTaker-pro dataset: this dataset is already in the repository in the folder [RuleTaker-pro Dataset](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleTaker-pro/Dataset)
- RuleTaker dataset: The main part of this dataset that we use during our training is already in the repository in the folder [RuleTaker-pro Dataset](https://github.com/HLR/Probabilistic_Logical_Reasoning/tree/main/RuleTaker-pro/RuleTaker)
- RACE RoBERTa model: Originally RuleTaker is trained with a RoBERTa that is finetuned on the RACE dataset. Download this model from [BASERACE](https://drive.google.com/file/d/1tm3eJSMhebsyaj4eIS_Nmga11XZNOiGs/view?usp=sharing) and unzip its content in a folder named `BASERACE` in the RuleTaker-pro repository. 

## Training

The training script `train.py` is used to train models. 

To train the baseline RuleTaker-pro use these parameters:
```
!cd RuleTaker-pro
!python train.py --batch_size 16 --cur_epoch 4 --lr 1e-5 --losstype SimpleCE --adverb --race
```
To train RuleTaker-pro with PCT use these parameters:
```
!cd RuleTaker-pro
!python train.py --batch_size 16 --cur_epoch 4 --lr 1e-5 --losstype SimpleCE --adverb --race --race --alpha 0.01
```

You can customize the training process using various command line arguments:

- `--cuda_number`: Specify the CUDA device number for training.
- `--depth`: Set the depth of training.
- `--alpha`: The PCT multiplier.
- `--batch_size`: Determine the batch size for training.
- `--cur_epoch`: Set the number of epochs for the training.
- `--learning_rate`: Define the learning rate for the optimizer.
- `--apply_PCT`: Enable the PCT method during training.
- `--race`: Preload race values.
- `--model_name`: Set a name for the saved model.
- `--adverb`: Decide whether to use adverbs in the training process.
- `--losstype`: Select the loss type for training.


## Utils

The repository includes utility scripts (`utils.py` and `PCT_utils.py`) for dataset preparation, accuracy calculation, seeding, and more, supporting the training process.
