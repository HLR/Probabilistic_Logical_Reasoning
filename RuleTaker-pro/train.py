from utils import create_main_dataset,flat_accuracy,seed_everything,format_time,get_loss_fucntion,calculate_loss,download_and_unzip_BACERACE
from PCT_utils import create_PCT_dataset
import torch, time,argparse
from transformers import AdamW,get_linear_schedule_with_warmup,AutoModelForSequenceClassification



seed_everything(42)

parser = argparse.ArgumentParser(description='train RuleTaker-pro')
parser.add_argument('--cuda', dest='cuda_number', default=3, help='cuda number to train the models on',type=int)
parser.add_argument('--depth', dest='depth', default=3, help='depth of training',type=int)
parser.add_argument('--alpha', dest='alpha', default=10, help='TODO',type=float)
parser.add_argument('--batch_size', dest='batch_size', default=16, help='batch size for neural network training',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=4, help='number of epochs you want your model to train on',type=int)
parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',type=float)
parser.add_argument('--apply_PCT', action='store_true', help='Use PCT method during training')
parser.add_argument('--race', dest='race', default=True, help='preload race values',type=bool)
parser.add_argument('--model_name', dest='model_name', default="defaultname", help='save the name of the model',type=str)
parser.add_argument('--adverb', dest='adverb', default=True, help='whther or not to use adverbs',type=bool)
parser.add_argument('--losstype', choices=["wBCE", "SimpleCE", "MSE", "L1loss"], default="SimpleCE", help="Loss type for training")
args = parser.parse_args()
args.apply_PCT=True

PCT_batch_size=int(args.batch_size/4)
download_and_unzip_BACERACE()

device = torch.device("cuda:"+str(args.cuda_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)

# read the data
train_dataloader=create_main_dataset(split="train",depth=args.depth,batchsize=args.batch_size)
val_dataloader=create_main_dataset(split="dev",depth=args.depth,batchsize=args.batch_size)
if args.apply_PCT:
    PCT_dataloader=create_PCT_dataset(args.depth, f"RuleTaker/original/depth-{args.depth}/train.jsonl", 
                                      f"RuleTaker/original/depth-{args.depth}/meta-train.jsonl", adverb=True)

# Load model
model_arch='roberta-large'
if args.race:
    print(download_and_unzip_BACERACE())
    model_arch = 'BASERACE'
model = AutoModelForSequenceClassification.from_pretrained(model_arch, num_labels=2)
model = model.to(device)
    
optimizer = AdamW(model.parameters(),lr=args.learning_rate,betas=[0.9, 0.98],eps=1e-7,weight_decay=0.1)
total_steps = len(train_dataloader) * args.cur_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps),num_training_steps=int((1 - 0.06) * total_steps))
loss_fct=get_loss_fucntion(args.losstype)
softm = torch.nn.Softmax(dim=1)

if args.apply_PCT:
    PCT_dataloader_iter=iter(PCT_dataloader)

for epoch_i in range(args.cur_epoch):

    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.cur_epoch))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0.0

    model.train()
    model.zero_grad()
    for step, batch in enumerate(train_dataloader):     
        b_input_ids,b_input_mask,b_labels,b_weights = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
        o = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        logits = o.logits
        loss=calculate_loss(args.losstype,loss_fct,logits,b_labels,b_weights)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        if args.apply_PCT and epoch_i>0:
            for pd_i in range(PCT_batch_size):
                try:
                    PCT_batch=next(PCT_dataloader_iter)
                except:
                    PCT_dataloader_iter=iter(PCT_dataloader)
                    PCT_batch=next(PCT_dataloader_iter)

                b_input_ids,b_input_mask,train_proba_PD,prev_connect,rule_proba = [PCT_batch[PCT_batch_i].to(device) for PCT_batch_i in range(5)]
                o = model(b_input_ids, attention_mask=b_input_mask)
                logits = softm(o.logits)
                PCT_loss=torch.Tensor([0]).to(device)
                for ii in range(len(prev_connect)):
                    if not prev_connect[ii]==-1:
                        v=torch.abs(logits[ii][1]-logits[0][1]*rule_proba[ii]/100).to(device)
                        PCT_loss+=v*args.alpha   
                PCT_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    model.save_pretrained("models/"+args.model_name+str(epoch_i))
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    total_eval_accuracy,total_eval_loss,total_conf_acc,nb_eval_steps = 0, 0.0, 0, 0
    for batch in val_dataloader:

        b_input_ids,b_input_mask,b_labels,b_weights = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
        with torch.no_grad():
            o = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        logits = o.logits
        loss=calculate_loss(args.losstype,loss_fct,logits,b_labels,b_weights)
        total_eval_loss += loss.item()
        total_eval_accuracy += flat_accuracy(logits.detach().cpu().numpy(), b_labels.to('cpu').numpy())
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

    print("  Accuracy: {}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)