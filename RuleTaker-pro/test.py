from utils import create_main_dataset,seed_everything
from PCT_utils import create_PCT_dataset
import torch,argparse
from transformers import AutoModelForSequenceClassification

seed_everything(42)

parser = argparse.ArgumentParser(description='Run Logical Rulebert')
parser.add_argument('--cuda', dest='cuda_number', default=3, help='cuda number to train the models on',type=int)
parser.add_argument('--alpha', dest='alpha', default=10, help='TODO',type=float)
parser.add_argument('--batch_size', dest='batch_size', default=16, help='batch size for neural network training',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=4, help='number of epochs you want your model to train on',type=int)
parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',type=float)
parser.add_argument('--apply_PCT', action='store_true', help='Use PCT method during training')
parser.add_argument('--race', dest='race', default=True, help='preload race values',type=bool)
parser.add_argument('--model_name', dest='model_name', default="defaultname", help='load the name of the model',type=str)
parser.add_argument('--adverb', dest='adverb', default=True, help='whther or not to use adverbs',type=bool)
args = parser.parse_args()


out_put_file=open(args.model_name+"_testi.txt","w")
include_first=False
model_arch=args.model_name

test_dataloader=create_main_dataset(split="test",depth=5,batchsize=args.batch_size)
PCT_dataloader=create_PCT_dataset(args.depth, f"RuleTaker/original/depth-5/test.jsonl", 
                                      f"RuleTaker/original/depth-5/meta-test.jsonl", adverb=True)
# Load model
device = torch.device("cuda:"+str(args.cude_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)
model = AutoModelForSequenceClassification.from_pretrained(model_arch+str(3), num_labels=2)
model = model.to(device)
model.eval()
        
softm = torch.nn.Softmax(dim=1)
total_step_loss=0
PCT_dataloader_iter=iter(PCT_dataloader)
alpha=args.alpha

t_PCT,ac_PCT1,ac_PCT10,ac_PCT25=0,0,0,0
t_PCT_main,ac_PCT1_main,ac_PCT10_main,ac_PCT25_main=0,0,0,0

average_proba=0

ac_,t_=[0,0,0,0,0,0,0],[[0,0] for or_i in range(6)]
bac_PCT,bt_PCT=[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
bac_,bt_=[[0,0] for or_i in range(6)],[[0,0] for or_i in range(6)]
ac25,ac10,ac1=[[0,0] for or_i in range(6)],[[0,0] for or_i in range(6)],[[0,0] for or_i in range(6)]
L1_list,MSE_list,MSEL1_T=[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[1,1,1,1,1,1,1]

for step, batch in enumerate(test_dataloader):     

    b_input_ids,b_input_mask,b_labels,b_weights,b_depths,or_used = [batch[batch_i].to(device) for batch_i in range(6)]
    with torch.no_grad():
        o = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
    logits = o.logits
    for hh_,yy_,dd_,or_i in zip(softm(logits),b_weights,b_depths,or_used):

        L1_list[dd_]+=abs(hh_[1]-yy_)
        MSE_list[dd_]+=(hh_[1]-yy_)**2
        MSEL1_T[dd_]+=1
        if (hh_[1]>0.5 and yy_>0.5) or (hh_[1]<0.5 and yy_<0.5):
            bac_[dd_][or_i]+=1
        if abs(hh_[1]-yy_)<0.25:
            ac25[dd_][or_i]+=1
        if abs(hh_[1]-yy_)<0.10:
            ac10[dd_][or_i]+=1
        if abs(hh_[1]-yy_)<0.01:
            ac1[dd_][or_i]+=1
        t_[dd_][or_i]+=1
        
        average_proba+=hh_[1]

for PCT_batch in PCT_dataloader_iter:

    b_input_ids,b_input_mask,train_proba_PCT,prev_connect,rule_proba,is_main_question = [PCT_batch[batch_i].to(device) for batch_i in range(6)]
    with torch.no_grad():
        o = model(b_input_ids, attention_mask=b_input_mask)
    logits = softm(o.logits)

    for ii in range(len(prev_connect)):
        if not prev_connect[ii]==-1:
            v=torch.abs(logits[ii][1]-logits[0][1]*rule_proba[ii]/100).to(device)
            if v<0.01:
                ac_PCT1+=1
                if is_main_question[ii]:
                    ac_PCT1_main+=1
            if v<0.10:
                ac_PCT10+=1
                if is_main_question[ii]:
                    ac_PCT10_main+=1
            if v<0.25:
                ac_PCT25+=1
                if is_main_question[ii]:
                    ac_PCT25_main+=1

            if is_main_question[ii]:
                t_PCT_main+=1
            t_PCT+=1
                


print("L1 losses for all depths",[i/it for i,it in zip(L1_list,MSEL1_T)])
print("Total L1 ",sum(L1_list)/sum(MSEL1_T))
print("MSE losses for all depths",[i/it for i,it in zip(MSE_list,MSEL1_T)])
print("Total MSE ",sum(MSE_list)/sum(MSEL1_T))
print("average of probablities:",average_proba/((sum([sum(i) for i in t_])+1)))

print("Total")
print("AC25",(sum([sum(i) for i in ac25])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac25])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac25])+1)/(sum([i[1] for i in t_])+1))
print("AC10",(sum([sum(i) for i in ac10])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac10])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac10])+1)/(sum([i[1] for i in t_])+1))
print("AC1",(sum([sum(i) for i in ac1])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac1])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac1])+1)/(sum([i[1] for i in t_])+1))
print("BA",(sum([sum(i) for i in bac_])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in bac_])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in bac_])+1)/(sum([i[1] for i in t_])+1))
for dd_ in range(0,6,1):
    print("depth: ",dd_)
    print("AC25",(ac25[dd_][0]+ac25[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac25[dd_][0]+1)/(t_[dd_][0]+1),(ac25[dd_][1]+1)/(t_[dd_][1]+1))
    print("AC10",(ac10[dd_][0]+ac10[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac10[dd_][0]+1)/(t_[dd_][0]+1),(ac10[dd_][1]+1)/(t_[dd_][1]+1))
    print("AC1",(ac1[dd_][0]+ac1[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac1[dd_][0]+1)/(t_[dd_][0]+1),(ac1[dd_][1]+1)/(t_[dd_][1]+1))
    print("BA",(bac_[dd_][0]+bac_[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(bac_[dd_][0]+1)/(t_[dd_][0]+1),(bac_[dd_][1]+1)/(t_[dd_][1]+1))

print(" ")
print("Main dataset questions CS1: ",(ac_PCT1_main+1)/(t_PCT_main+1))
print("Main dataset questions CS10: ",(ac_PCT10_main+1)/(t_PCT_main+1))
print("Main dataset questions CS25: ",(ac_PCT25_main+1)/(t_PCT_main+1))

print(" ")
print("All questions CS1: ",(ac_PCT1+1)/(t_PCT+1))
print("All questions CS10: ",(ac_PCT10+1)/(t_PCT+1))
print("All questions CS25: ",(ac_PCT25+1)/(t_PCT+1))


print("L1 losses for all depths",[i/it for i,it in zip(L1_list,MSEL1_T)],file=out_put_file)
print("Total L1 ",sum(L1_list)/sum(MSEL1_T),file=out_put_file)
print("MSE losses for all depths",[i/it for i,it in zip(MSE_list,MSEL1_T)],file=out_put_file)
print("Total MSE ",sum(MSE_list)/sum(MSEL1_T),file=out_put_file)
print("average of probablities:",average_proba/((sum([sum(i) for i in t_])+1)),file=out_put_file)

print("Total",file=out_put_file)
print("AC25",(sum([sum(i) for i in ac25])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac25])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac25])+1)/(sum([i[1] for i in t_])+1),file=out_put_file)
print("AC10",(sum([sum(i) for i in ac10])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac10])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac10])+1)/(sum([i[1] for i in t_])+1),file=out_put_file)
print("AC1",(sum([sum(i) for i in ac1])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in ac1])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in ac1])+1)/(sum([i[1] for i in t_])+1),file=out_put_file)
print("BA",(sum([sum(i) for i in bac_])+1)/(sum([sum(i) for i in t_])+1),(sum([i[0] for i in bac_])+1)/(sum([i[0] for i in t_])+1),(sum([i[1] for i in bac_])+1)/(sum([i[1] for i in t_])+1),file=out_put_file)
for dd_ in range(0,6,1):
    print("depth: ",dd_,file=out_put_file)
    print("AC25",(ac25[dd_][0]+ac25[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac25[dd_][0]+1)/(t_[dd_][0]+1),(ac25[dd_][1]+1)/(t_[dd_][1]+1),file=out_put_file)
    print("AC10",(ac10[dd_][0]+ac10[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac10[dd_][0]+1)/(t_[dd_][0]+1),(ac10[dd_][1]+1)/(t_[dd_][1]+1),file=out_put_file)
    print("AC1",(ac1[dd_][0]+ac1[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(ac1[dd_][0]+1)/(t_[dd_][0]+1),(ac1[dd_][1]+1)/(t_[dd_][1]+1),file=out_put_file)
    print("BA",(bac_[dd_][0]+bac_[dd_][1]+1)/(t_[dd_][0]+t_[dd_][1]+1),(bac_[dd_][0]+1)/(t_[dd_][0]+1),(bac_[dd_][1]+1)/(t_[dd_][1]+1),file=out_put_file)

print(" ",file=out_put_file)
print("Main dataset questions CS1: ",(ac_PCT1_main+1)/(t_PCT_main+1),file=out_put_file)
print("Main dataset questions CS10: ",(ac_PCT10_main+1)/(t_PCT_main+1),file=out_put_file)
print("Main dataset questions CS25: ",(ac_PCT25_main+1)/(t_PCT_main+1),file=out_put_file)

print(" ",file=out_put_file)
print("All questions CS1: ",(ac_PCT1+1)/(t_PCT+1),file=out_put_file)
print("All questions CS10: ",(ac_PCT10+1)/(t_PCT+1),file=out_put_file)
print("All questions CS25: ",(ac_PCT25+1)/(t_PCT+1),file=out_put_file)

out_put_file.flush()
