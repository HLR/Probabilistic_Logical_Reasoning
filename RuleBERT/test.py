import torch,json,argparse
from transformers import AutoModelForSequenceClassification
from utils import flat_accuracy,seed_everything
from dataset_preprocessing import return_torch_datasets

parser = argparse.ArgumentParser(description='Testing models of RuleBERT')
parser.add_argument('--seed', type=int, default=42, help='Random seed initializer')
parser.add_argument('--cuda_number', type=int, default=6, help='CUDA number for GPU testing')
parser.add_argument('--use_ruletext', action='store_true', help='Use rule text during testing')
parser.add_argument('--model_name', type=str, default="default_name", help='Model load name')
parser.add_argument('--epoch', type=int, default=2, help='epoch of the saved model to be tested')
parser.add_argument('--data_dir', type=str, default="data/", help='Data directory')
args = parser.parse_args()

seed_everything(args.seed)

include_first=True
model_arch = 'roberta-large'
max_length = 512
batch_size = 128
device = torch.device("cuda:"+str(args.cuda_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)

for depths_ in range(1,6):
    print("depths_:",depths_)
    test_dataloader,_,test_dataloader_PD,_,rule_lambda = return_torch_datasets("TestingData/Depth"+str(depths_)+".jsonl",include_first,args.use_ruletext,model_arch,max_length,batch_size,test=True)
    model = AutoModelForSequenceClassification.from_pretrained("models/"+args.model_name+" epoch "+str(args.epoch))
    model = model.to(device)
    model.eval()

    # ========================================
    #               Testing
    # ========================================
    total_eval_accuracy=0
    for step, batch in enumerate(test_dataloader):
        b_input_ids, b_input_mask, b_labels, b_weights = [batch[b_num].to(device) for b_num in range(4)]
        with torch.no_grad():
            o = model(b_input_ids,attention_mask=b_input_mask)
        total_eval_accuracy += flat_accuracy(o.logits.detach().cpu().numpy(), b_labels.to('cpu').numpy())
    avg_val_conf_acc = total_eval_accuracy / len(test_dataloader)
    print(" Binary Accuracy: {}".format(avg_val_conf_acc))
    if depths_==5:
        CS1,CS10,CS25,CST=0,0,0,0
        for step_PD, pd_batch_number in enumerate(test_dataloader_PD):

            b_input_ids,b_input_mask,b_questions_spots,answer_probabilities,b_rule_number = [pd_batch_number[b_num].to(device) for b_num in range(5)]
            with torch.no_grad():
                o = model(b_input_ids, attention_mask=b_input_mask)
            logits = torch.nn.functional.softmax(o.logits, dim=1)
            v=torch.Tensor([1]).to(device)
            for ii in range(len(b_questions_spots)):
                if b_questions_spots[ii]:
                    v=torch.abs(logits[ii][0]-v*answer_probabilities[ii])
                    if v.item()<=0.01:
                        CS1+=1
                    if v.item()<=0.1:
                        CS10+=1
                    if v.item()<=0.25:
                        CS25+=1
                    CST+=1
                    v=torch.Tensor([1]).to(device)
                else:
                    v=v*logits[ii][1]
        print("PCT CS1 : ",CS1/CST)
        print("PCT CS10 : ",CS10/CST)
        print("PCT CS25 : ",CS25/CST)