import torch,json
from typing import Any, Dict, List, cast
from random import sample
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import  AutoTokenizer
from tqdm import tqdm
from copy import deepcopy
from utils import make_new_sentecnes

def return_torch_datasets(data_dir,include_first,use_ruletext,model_arch,max_length,batch_size,test=False):
    tokenizer = AutoTokenizer.from_pretrained(model_arch)
    if test:
        train_file=data_dir
        val_file=data_dir
    else:
        train_file = data_dir + 'train.jsonl'
        val_file = data_dir + 'val.jsonl'
    rule_lambda=dict()

    val_theories = [json.loads(jline) for jline in open(val_file, "r").read().splitlines()]
    if use_ruletext:
        val_context = [t['context'] for t in val_theories]
    else:
        val_context = [t['facts_sentence'] for t in val_theories]
    val_hypotheses = [t['hypothesis_sentence'] for t in val_theories]
    val_rules_PD = [make_new_sentecnes(t,include_first,rule_lambda) for t in val_theories]

    val_labels_ = [1 if t['output'] else 0 for t in val_theories]
    val_data_weights_ = [t['hyp_weight'] for t in val_theories]
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

    val_data_weights = torch.tensor(val_data_weights_)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels, val_data_weights)
    val_dataset_PD = TensorDataset(val_input_ids_PD, val_attention_masks_PD,torch.LongTensor(val_questions_spots_PD),torch.Tensor(val_answer_probabilities_PD),torch.LongTensor(val_answer_rule_number_PD))

    


    val_dataloader = DataLoader(dataset=val_dataset,
                                sampler=SequentialSampler(val_dataset),
                                batch_size=batch_size,
                                )

    val_dataloader_PD = DataLoader(dataset=val_dataset_PD,
                                sampler=SequentialSampler(val_dataset_PD),
                                batch_size=batch_size,
                                )
    if test:
        return val_dataloader,None,val_dataloader_PD,None,rule_lambda
    
    train_theories_1 = [json.loads(jline) for jline in open(train_file, "r").read().splitlines()]
    for x in tqdm(train_theories_1):
        if(not x['output']):
            x['hyp_weight'] = 1 - x['hyp_weight']

    train_theories_1 = sample(train_theories_1, len(train_theories_1))
    train_theories_2 = deepcopy(train_theories_1)
    for x in tqdm(train_theories_2):
        x['output'] = False if x['output'] else True
        x['hyp_weight'] = 1 - x['hyp_weight']

    train_theories = cast(List[Dict[Any, Any]], [item for sublist
                        in list(map(list, zip(train_theories_1, train_theories_2))) for item in sublist])

    if use_ruletext:
        train_context = [t['context'] for t in train_theories]
    else:
        train_context = [t['facts_sentence'] for t in train_theories]
    train_hypotheses = [t['hypothesis_sentence'] for t in train_theories]
    train_rules_PD = [make_new_sentecnes(t,include_first,rule_lambda) for t in train_theories]

    train_labels_ = [1 if t['output'] else 0 for t in train_theories]

    train_data_weights_ = [t['hyp_weight'] for t in train_theories]
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

    train_data_weights = torch.tensor(train_data_weights_)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_data_weights)
    train_dataset_PD = TensorDataset(train_input_ids_PD, train_attention_masks_PD,torch.LongTensor(train_questions_spots_PD),torch.Tensor(train_answer_probabilities_PD),torch.LongTensor(train_answer_rule_number_PD))
    train_dataloader = DataLoader(dataset=train_dataset,
                                sampler=SequentialSampler(train_dataset),
                                batch_size=batch_size,
                                )
    train_dataloader_PD = DataLoader(dataset=train_dataset_PD,
                                sampler=SequentialSampler(train_dataset_PD),
                                batch_size=batch_size,
                                )

    return train_dataloader,val_dataloader,train_dataloader_PD,val_dataloader_PD,rule_lambda