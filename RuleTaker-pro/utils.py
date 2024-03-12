import random, os, torch, datetime, torch, tqdm,subprocess, numpy as np, pandas as pd,requests,tarfile
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import  AutoTokenizer
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def create_main_dataset(split="train",depth=3,batchsize=16, adverb=True):
    seed_everything(42)
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    model_arch = 'roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_arch)
    dataset=pd.read_csv(f"dataset/d{depth}/{split}-pro.csv")
    input_ids,attention_masks=[],[]
    for c, h in tqdm.tqdm(zip(dataset["context"], dataset["question"])):
        encoded = tokenizer.encode_plus(c, h, max_length=384, truncation=True, return_tensors='pt',padding='max_length')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    tensor_dataset = TensorDataset(input_ids, attention_masks, torch.LongTensor(dataset["label"]), torch.FloatTensor(dataset["proba"]), torch.LongTensor(dataset["depth"]),torch.LongTensor(dataset["complex"]))
    dataloader = DataLoader(dataset=tensor_dataset, sampler=RandomSampler(tensor_dataset), batch_size=batchsize)
    return dataloader

def get_loss_fucntion(losstype):
    if losstype=="wBCE":
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    elif losstype=="SimpleCE":
        loss_fct = loss_fct = torch.nn.CrossEntropyLoss()
    elif losstype=="MSE":    
        loss_fct = torch.nn.MSELoss()
    elif losstype=="L1loss":    
        loss_fct = torch.nn.L1Loss()
    return loss_fct

def calculate_loss(losstype,loss_fct,logits,b_labels,b_weights):
    if losstype=="wBCE":
        loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)
    elif losstype=="SimpleCE":
        loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
    elif losstype=="MSE":    
        loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)
    elif losstype=="L1loss":    
        loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)
    return loss



def download_and_unzip_BACERACE():
    if os.path.exists("BASERACE"):
        return "BASERACE exists."

    return """Please download the BASERACE model from the link:
    https://drive.google.com/file/d/1tm3eJSMhebsyaj4eIS_Nmga11XZNOiGs/view?usp=sharing
    and unzip it in the folder named BASERACE in the same directory"""

