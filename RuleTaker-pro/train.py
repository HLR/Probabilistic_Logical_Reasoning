import os

from utils import create_data_set,flat_accuracy,seed_everything,format_time,create_data_set_proba
from typing import Any, Dict, List, cast
import torch,json
from random import sample
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from copy import deepcopy
import time
import inspect
import argparse

seed_everything(42)

parser = argparse.ArgumentParser(description='Run Logical Rulebert')
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--chain', dest='chain_number', default=3, help='TODO',type=int)
parser.add_argument('--alpha', dest='alpha', default=10, help='TODO',type=float)
parser.add_argument('--batch', dest='batch_size', default=16, help='batch size for neural network training',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=6, help='number of epochs you want your model to train on',type=int)
parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',type=float)
parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--race', dest='race', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--samplenum', dest='samplenum', default=1000000000, help='number of samples to train the model on',type=int)
parser.add_argument('--model_name', dest='model_name', default="modellogical", help='TODO',type=str)
parser.add_argument('--context', dest='context', default=False, help='TODO',type=bool)
parser.add_argument('--adverb', dest='adverb', default=False, help='TODO',type=bool)
parser.add_argument('--deberta', dest='deberta', default=False, help='TODO',type=bool)
parser.add_argument('--fake', dest='fake', default=False, help='TODO',type=bool)
parser.add_argument('--mustrule', dest='mustrule', default=False, help='TODO',type=bool)
parser.add_argument('--losstype', dest='losstype', default="CE", help='the loss type to messure the gradients',type=str)
parser.add_argument('--logloss', dest='logloss', default=False, help='TODO',type=bool)
args = parser.parse_args()



cude_number=args.cuda_number
chain_number=args.chain_number
batch_size = args.batch_size
epochs = args.cur_epoch
lr = args.learning_rate
apply_PD=args.primaldual
samplenum=args.samplenum
adverb=args.adverb
mustrule=args.mustrule

batch_size_real=4
batch_size_divided=batch_size/batch_size_real

if args.context:
    use_context=True
else:
    use_context=False
out_put_file=open(args.model_name+".txt","w")

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
for i in [cude_number,chain_number,batch_size,epochs,lr,apply_PD,samplenum,out_put_file,use_context,args.race,mustrule]:
    print(retrieve_name(i),i)
    print(retrieve_name(i),i,file=out_put_file)
    out_put_file.flush()

include_first=False
data_dir = "/home/ruletaker/rule-reasoning-dataset-V2020.2.5.0/original/depth-"+str(chain_number)+"/"
if args.race:
    model_arch = 'BASERACE'
elif args.deberta:
    model_arch="microsoft/deberta-large"
else:
    model_arch = "microsoft/deberta-v2-xlarge"
    
max_length = 384
eps = 1e-6
weight_decay =  0.1
warmup_ratio = 0.06
verbose = True
time_step_size = 10000

device = torch.device("cuda:"+str(cude_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)

# read the data

train_dataloader,train_dataloader_PD=create_data_set_proba(data_dir+"train.jsonl",data_dir+"meta-train.jsonl",samplenum,batch_size_real,adverb,args.fake,chain_number,mustrule)
val_dataloader,val_dataloader_PD=create_data_set_proba(data_dir+"dev.jsonl",data_dir+"meta-dev.jsonl",samplenum,batch_size_real,adverb,args.fake,chain_number,mustrule)

# Load model
print("DEBUG",device)

model = AutoModelForSequenceClassification.from_pretrained(model_arch, num_labels=2)
model = model.to(device)
    
optimizer = AdamW(model.parameters(),lr=lr,betas=[0.9, 0.98],eps=eps,weight_decay=weight_decay)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps),num_training_steps=int((1 - warmup_ratio) * total_steps))
if args.losstype=="CE":
    loss_fct = CrossEntropyLoss(reduction='none')
elif args.losstype=="SimpleCE":
    loss_fct = loss_fct = CrossEntropyLoss()
elif args.losstype=="MSE":    
    loss_fct = torch.nn.MSELoss()
elif args.losstype=="L1loss":    
    loss_fct = torch.nn.L1Loss()


    
training_stats = []
total_t0 = time.time()
from torch import nn
softm = nn.Softmax(dim=1)
total_step_loss=0
train_dataloader_PD_iter=iter(train_dataloader_PD)
alpha=args.alpha
for epoch_i in range(epochs):

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
    for step, batch in enumerate(train_dataloader):     

        if step % time_step_size == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            if verbose:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                print(total_step_loss/time_step_size)
                
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),file=out_put_file)
                print(total_step_loss/time_step_size,file=out_put_file)
                out_put_file.flush()
                
                total_step_loss=0

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_weights = batch[3].to(device)
        
        o = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
 
        logits = o.logits
        if args.losstype=="CE":
            loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)
        elif args.losstype=="SimpleCE":
            loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
        elif args.losstype=="MSE":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)
        elif args.losstype=="L1loss":    
            loss = loss_fct(logits.view(-1, 2).softmax(dim=1)[:,1], b_weights)



        total_train_loss += loss.item()
        total_step_loss += loss.item()
        loss.backward()
        if step==17000:
            model.save_pretrained(args.model_name+str(epoch_i)+".5")
        if step%batch_size_divided==batch_size_divided-1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if apply_PD and step%4==3 and epoch_i>0:
            for pd_i in range(4):
                try:
                    pd_batch_number=next(train_dataloader_PD_iter)
                except:
                    train_dataloader_PD_iter=iter(train_dataloader_PD)
                    pd_batch_number=next(train_dataloader_PD_iter)

                b_input_ids = pd_batch_number[0].to(device)
                b_input_mask = pd_batch_number[1].to(device)
                train_proba_PD = pd_batch_number[2].to(device)
                prev_connect = pd_batch_number[3].to(device)
                rule_proba = pd_batch_number[4].to(device)

                o = model(b_input_ids, attention_mask=b_input_mask)
                logits = softm(o.logits)

                loss2=torch.Tensor([0]).to(device)
                flag=False
                for ii in range(len(prev_connect)):
                    if not prev_connect[ii]==-1:
                        if args.logloss:
                            
                            v=torch.abs(logits[ii][1]-logits[0][1]*rule_proba[ii]/100).to(device)+torch.abs(torch.log(logits[ii][1])-torch.log(logits[0][1])-torch.log(rule_proba[ii]/100)).to(device)
                            
                        else:
                            
                            v=torch.abs(logits[ii][1]-logits[0][1]*rule_proba[ii]/100).to(device)
                        
                        loss2+=v*alpha
                        
                        flag=True    
                        #print(loss2,alpha)
                loss2.backward()
            #print(loss2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    if verbose:
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        print("  Average training loss: {0:.2f}".format(avg_train_loss),file=out_put_file)
        print("  Training epcoh took: {:}".format(training_time),file=out_put_file)
        out_put_file.flush()
        
    model.save_pretrained(args.model_name+str(epoch_i))
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
        b_weights = batch[3].to(device)
        
        with torch.no_grad():

            o = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)


        logits = o.logits
        loss = torch.mean(loss_fct(logits.view(-1, 2), b_labels.view(-1)) * b_weights)
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)


    print("  Accuracy: {}".format(avg_val_accuracy))
    print("  Accuracy: {}".format(avg_val_accuracy),file=out_put_file)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss),file=out_put_file)
    print("  Validation took: {:}".format(validation_time),file=out_put_file)
    out_put_file.flush()
