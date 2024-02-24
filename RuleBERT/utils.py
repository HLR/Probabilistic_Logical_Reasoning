import random, torch,datetime,re 
import numpy as np
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
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
    
def make_relation_to_str(x1,x2,x3):
    if x1[:3]=="neg":
        return "The "+x1[3:]+" of "+x2+" is not "+x3+"."
    return "The "+x1+" of "+x2+" is "+x3+"."

def make_new_sentecnes(example,include_first,rule_lambda):
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

def load_loss_function(losstype):
    loss_fct = CrossEntropyLoss(reduction='none')
    if losstype=="wBCE":
        loss_fct = CrossEntropyLoss(reduction='none')
    elif losstype=="SimpleCE":
        loss_fct = loss_fct = CrossEntropyLoss()
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