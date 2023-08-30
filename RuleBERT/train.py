
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
from scipy.special import softmax
import datetime
import numpy as np
seed_everything(42)
from typing import Any, Dict, List, cast
import torch
import json
from random import sample
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
# from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from copy import deepcopy
import time

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def confidence_accuracy(logits, labels, weights, threshold=0.05, verbose=False):
    # probs = torch.nn.functional.softmax(logits, dim=1)
    probs = softmax(logits, axis=1)
    # pred_weights = np.array([x[l].item() for x, l in zip(probs, labels)])
    pred_weights = np.array([x[1].item() for x in probs])  # get true prob
    abs_diff = np.abs(pred_weights - weights.cpu().numpy())
    if not verbose:
        return np.sum(abs_diff < threshold) / len(abs_diff)
    else:
        return probs, abs_diff

include_first=False
chain_number=3 ##
cude_number=2 ##
apply_PD=False ##
use_context=False ##
losstype="CE"
model_name_saved="Original_rubert_withouttext_3" ##
data_dir = "/home/data/chain_rules/Depth_"+str(chain_number)+"/"
model_arch = 'roberta-large'
max_length = 512
batch_size = 16
lr = 1e-6
eps = 2e-9 
weight_decay =  0.1
epochs = 4

warmup_ratio = 0.06
verbose = True
hard_rule = False
time_step_size = 1000


use_adverbs_of_frequency=False
use_numbers_for_frequency=False


device = torch.device("cuda:"+str(cude_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)

train_file = data_dir + 'train.jsonl'
val_file = data_dir + 'val.jsonl'

train_theories_1 = [json.loads(jline) for jline in open(train_file, "r").read().splitlines()]
val_theories = [json.loads(jline) for jline in open(val_file, "r").read().splitlines()]

alpha=0.1

import re,random

rule_lambda=dict()
rule_lambda_counter=1

def make_relation_to_str(x1,x2,x3):
    if x1[:3]=="neg":
        return "The "+x1[3:]+" of "+x2+" is not "+x3+"."
    return "The "+x1+" of "+x2+" is "+x3+"."

def make_new_sentecnes(example):
    all_new_sentences=[]
    all_new_sentences_relations=[]
    all_new_sentences_probablities=[]
    all_new_sentences_rule_number=[]
    
    if "evidence" in example:
        for ev_num,i in enumerate(example["evidence"].split("\n")):

            rule=re.findall("[a-zA-Z_]+",i.split(".")[0])
            facts=re.findall("[a-zA-Z_]+"," ".join(i.split(".")[1:-1]))
            A=rule[3:].index("A")
            B=rule[3:].index("B")
            head=rule[0]+'''("{}","{}")'''.format(facts[A],facts[B])
            head_sentence=make_relation_to_str(rule[0],facts[A],facts[B])
            
            probability_of_head=float(example["solution"][head])
            rule_sentences=[make_relation_to_str(facts[i],facts[i+1],facts[i+2]) for i in range(0,len(facts),3)]
            
            rule_string=rule[0]+" ".join([facts[i] for i in range(0,len(facts),3)])
            if not rule_string in rule_lambda:
                rule_lambda[rule_string]=len(rule_lambda)
            
            if include_first or ev_num>0:
                all_new_sentences.extend(rule_sentences)
                all_new_sentences_relations.extend([0 for i in range(len(rule_sentences))])
                all_new_sentences_probablities.extend([-1 for i in range(len(rule_sentences))])
                all_new_sentences_rule_number.extend([-1 for i in range(len(rule_sentences))])

                all_new_sentences.append(head_sentence)
                all_new_sentences_relations.append(1)
                all_new_sentences_probablities.append(float(example["rule_support"][ev_num]))
                all_new_sentences_rule_number.append(rule_lambda[rule_string])

    return all_new_sentences,all_new_sentences_relations,all_new_sentences_probablities,all_new_sentences_rule_number
if not hard_rule:
    for x in tqdm(train_theories_1):
        if(not x['output']):
            x['hyp_weight'] = 1 - x['hyp_weight']

train_theories_1 = sample(train_theories_1, len(train_theories_1))

if not hard_rule:

    train_theories_2 = deepcopy(train_theories_1)
    for x in tqdm(train_theories_2):
        x['output'] = False if x['output'] else True
        x['hyp_weight'] = 1 - x['hyp_weight']

    train_theories = cast(List[Dict[Any, Any]], [item for sublist
                        in list(map(list, zip(train_theories_1, train_theories_2))) for item in sublist])
else:
    train_theories = train_theories_1

# prepare training data
if use_context:
    train_context = [t['context'] for t in train_theories]
else:
    train_context = [t['facts_sentence'] for t in train_theories]
train_hypotheses = [t['hypothesis_sentence'] for t in train_theories]
train_rules_PD = [make_new_sentecnes(t) for t in train_theories]

train_labels_ = [1 if t['output'] else 0 for t in train_theories]
if not hard_rule:
    train_data_weights_ = [t['hyp_weight'] for t in train_theories]

# prepare val data
if use_context:
    val_context = [t['context'] for t in val_theories]
else:
    val_context = [t['facts_sentence'] for t in val_theories]
val_hypotheses = [t['hypothesis_sentence'] for t in val_theories]
val_rules_PD = [make_new_sentecnes(t) for t in val_theories]

val_labels_ = [1 if t['output'] else 0 for t in val_theories]
if not hard_rule:
    val_data_weights_ = [t['hyp_weight'] for t in val_theories]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

# tokenize training data
train_input_ids_ = []
train_attention_masks_ = []

for c, h in tqdm(zip(train_context, train_hypotheses)):
    encoded = tokenizer.encode_plus(c, h,
                                    max_length=max_length,
                                    truncation=True,
                                    return_tensors='pt',
                                    padding='max_length')
    train_input_ids_.append(encoded['input_ids'])
    train_attention_masks_.append(encoded['attention_mask'])

train_input_ids = torch.cat(train_input_ids_, dim=0)
train_attention_masks = torch.cat(train_attention_masks_, dim=0)

train_labels = torch.tensor(train_labels_)

# tokenize training data
batch_flow=0
train_input_ids_PD = []
train_attention_masks_PD = []
train_questions_spots_PD=[]
train_answer_probabilities_PD=[]
train_answer_rule_number_PD=[]

temp_filler=None
for c, h in tqdm(zip(train_context[::2], train_rules_PD[::2])):
    if batch_flow+len(h[0])>batch_size:
        for j in range(batch_size-batch_flow):

            train_input_ids_PD.append(temp_filler)
            train_attention_masks_PD.append(temp_filler)
            train_questions_spots_PD.append(0)
            train_answer_probabilities_PD.append(0)
            train_answer_rule_number_PD.append(-1)
        batch_flow=0
        
    for sent_id in range(len(h[0])):
        batch_flow+=1
        batch_flow%=batch_size
        encoded = tokenizer.encode_plus(c ,h[0][sent_id],
                                        max_length=max_length,
                                        truncation=True,
                                        return_tensors='pt',
                                        padding='max_length')

        train_input_ids_PD.append(encoded['input_ids'])
        train_attention_masks_PD.append(encoded['attention_mask'])
        temp_filler=encoded['attention_mask']
    
        train_questions_spots_PD.append(h[1][sent_id])
        train_answer_probabilities_PD.append(h[2][sent_id])
        train_answer_rule_number_PD.append(h[3][sent_id])


train_input_ids_PD = torch.cat(train_input_ids_PD, dim=0)
train_attention_masks_PD = torch.cat(train_attention_masks_PD, dim=0)

train_labels = torch.tensor(train_labels_)
if not hard_rule:
    train_data_weights = torch.tensor(train_data_weights_)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_data_weights)
    train_dataset_PD = TensorDataset(train_input_ids_PD, train_attention_masks_PD,torch.LongTensor(train_questions_spots_PD),torch.Tensor(train_answer_probabilities_PD),torch.LongTensor(train_answer_rule_number_PD))
else:
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    
#print(train_dataset[0])
#print(train_dataset_PD[0])

# tokenize val data
val_input_ids_ = []
val_attention_masks_ = []

for c, h in tqdm(zip(val_context, val_hypotheses)):
    encoded = tokenizer.encode_plus(c, h,
                                    max_length=max_length,
                                    truncation=True,
                                    return_tensors='pt',
                                    padding='max_length')
    val_input_ids_.append(encoded['input_ids'])
    val_attention_masks_.append(encoded['attention_mask'])

val_input_ids = torch.cat(val_input_ids_, dim=0)
val_attention_masks = torch.cat(val_attention_masks_, dim=0)

val_labels = torch.tensor(val_labels_)

# tokenize val data
batch_flow=0
val_input_ids_PD = []
val_attention_masks_PD = []
val_questions_spots_PD=[]
val_answer_probabilities_PD=[]
val_answer_rule_number_PD=[]

temp_filler=None
for c, h in tqdm(zip(val_context, val_rules_PD)):
    
    if batch_flow+len(h[0])>batch_size:
        for j in range(batch_size-batch_flow):
        
            val_input_ids_PD.append(temp_filler)
            val_attention_masks_PD.append(temp_filler)
            val_questions_spots_PD.append(0)
            val_answer_probabilities_PD.append(0)
            val_answer_rule_number_PD.append(-1)
        batch_flow=0
        
    for sent_id in range(len(h[0])):
        batch_flow+=1
        batch_flow%=batch_size
        
        encoded = tokenizer.encode_plus(c ,h[0][sent_id],
                                        max_length=max_length,
                                        truncation=True,
                                        return_tensors='pt',
                                        padding='max_length')

        val_input_ids_PD.append(encoded['input_ids'])
        val_attention_masks_PD.append(encoded['attention_mask'])
        temp_filler=encoded['attention_mask']
    
        val_questions_spots_PD.append(h[1][sent_id])
        val_answer_probabilities_PD.append(h[2][sent_id])
        val_answer_rule_number_PD.append(h[3][sent_id])

val_input_ids_PD = torch.cat(val_input_ids_PD, dim=0)
val_attention_masks_PD = torch.cat(val_attention_masks_PD, dim=0)

val_labels = torch.tensor(val_labels_)
if not hard_rule:
    val_data_weights = torch.tensor(val_data_weights_)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels, val_data_weights)
    val_dataset_PD = TensorDataset(val_input_ids_PD, val_attention_masks_PD,torch.LongTensor(val_questions_spots_PD),torch.Tensor(val_answer_probabilities_PD),torch.LongTensor(val_answer_rule_number_PD))
else:
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
   
train_dataloader = DataLoader(dataset=train_dataset,
                              sampler=SequentialSampler(train_dataset),
                              batch_size=batch_size,
                              )

val_dataloader = DataLoader(dataset=val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=batch_size,
                            )

train_dataloader_PD = DataLoader(dataset=train_dataset_PD,
                              sampler=SequentialSampler(train_dataset_PD),
                              batch_size=batch_size,
                              )

val_dataloader_PD = DataLoader(dataset=val_dataset_PD,
                            sampler=SequentialSampler(val_dataset_PD),
                            batch_size=batch_size,
                            )

print(len(train_dataloader),len(train_dataloader_PD))
print(len(val_dataloader),len(val_dataloader_PD))
lambdas=[0 for i in range(len(rule_lambda))]

# Load model
print("DEBUG",device)

model = AutoModelForSequenceClassification.from_pretrained(model_arch, num_labels=2)
model = model.to(device)
for name , param in list(model.named_parameters())[:-72]:
    param.requires_grad = False

optimizer = AdamW(model.parameters(),lr=lr,eps=eps,weight_decay=weight_decay)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps),num_training_steps=int((1 - warmup_ratio) * total_steps))
loss_fct = CrossEntropyLoss(reduction='none')
if losstype=="CE":
    loss_fct = CrossEntropyLoss(reduction='none')
elif losstype=="SimpleCE":
    loss_fct = loss_fct = CrossEntropyLoss()
elif losstype=="MSE":    
    loss_fct = torch.nn.MSELoss()
elif losstype=="L1loss":    
    loss_fct = torch.nn.L1Loss()
training_stats = []
total_t0 = time.time()
from torch import nn
softm = nn.Softmax(dim=1)
train_dataloader_PD_iter=iter(train_dataloader_PD)
for epoch_i in range(epochs):
    alpha*=0.9
    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0.0

    model.train()
    model.zero_grad()
    all_steps=len(train_dataloader)//3
    for step, batch in enumerate(train_dataloader):
        if step % all_steps == all_steps-1:
            model.save_pretrained(model_name_saved+str(epoch_i)+"_step_"+str(step))
        if step % time_step_size == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            if verbose:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        if not hard_rule:
            b_weights = batch[3].to(device)

        

        if not hard_rule:
            o = model(b_input_ids,
                    attention_mask=b_input_mask)
        else:
            o = model(b_input_ids, 
                      attention_mask=b_input_mask, 
                      labels=b_labels)
 
        logits = o.logits

        if losstype=="CE":
            loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)
        elif losstype=="SimpleCE":
            loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
        elif losstype=="MSE":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)
        elif losstype=="L1loss":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        if epoch_i>0 and apply_PD:
            try:
                pd_batch_number=next(train_dataloader_PD_iter)
            except:
                train_dataloader_PD_iter=iter(train_dataloader_PD)
                pd_batch_number=next(train_dataloader_PD_iter)

            b_input_ids = pd_batch_number[0].to(device)
            b_input_mask = pd_batch_number[1].to(device)
            b_questions_spots = pd_batch_number[2].to(device)
            answer_probabilities = pd_batch_number[3].to(device)
            b_rule_number = pd_batch_number[4].to(device)

            o = model(b_input_ids, attention_mask=b_input_mask)
            logits = softm(o.logits)

            loss2=torch.Tensor([0]).to(device)
            v=torch.Tensor([1]).to(device)
            for ii in range(len(b_questions_spots)):
                if b_questions_spots[ii]:
                    v=torch.abs(logits[ii][0]-v*answer_probabilities[ii])
                    loss2+=v*lambdas[b_rule_number[ii]]
                    lambdas[b_rule_number[ii]]+=alpha*v.item()
                    v=torch.Tensor([1]).to(device)
                else:
                    v=v*logits[ii][1]


            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
        

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)
    if verbose:
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
    model.save_pretrained(model_name_saved+str(epoch_i))
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0.0
    total_conf_acc = 0
    nb_eval_steps = 0

    for batch in val_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        if not hard_rule:
            b_weights = batch[3].to(device)

        with torch.no_grad():
            if not hard_rule:
                o = model(b_input_ids, attention_mask=b_input_mask)
            else:
                o = model(b_input_ids, 
                          attention_mask=b_input_mask, 
                          labels=b_labels)


        logits = o.logits
        if losstype=="CE":
            loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)
        elif losstype=="SimpleCE":
            loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
        elif losstype=="MSE":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)
        elif losstype=="L1loss":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

        if not hard_rule:
            total_conf_acc += confidence_accuracy(logits, b_labels, b_weights)
            avg_val_conf_acc = total_conf_acc / len(val_dataloader)

    print("  Accuracy: {}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time        
        }
    )

    if not hard_rule:
        training_stats.append(
            {
                'Val_Conf_Acc': avg_val_conf_acc
            }
        )

total_train_time = format_time(time.time() - total_t0)
training_stats.append({'total_train_time': total_train_time})
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(total_train_time))

model_arch = 'roberta-large'

training_stats.append({'hyperparameters': {'max_length': max_length,
                                           'batch_size': batch_size,
                                           'learning_rate': lr,
                                           'epsilon': eps,
                                           'weight_decay': weight_decay,
                                           'n_epochs': epochs,
                                           'warmup_ratio': warmup_ratio}})
training_stats.append({'model': model_arch,
                       'dataset': data_dir})

# output model and dict of results
model_path = f'models/{time.strftime("%Y%m%dT%H%M%S")}/'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
json.dump(training_stats, open(f"{model_path}train_stats.json", "w"), indent=4)

