import time, torch,argparse
from transformers import AdamW,get_linear_schedule_with_warmup,AutoModelForSequenceClassification, AutoTokenizer
from utils import seed_everything, format_time, flat_accuracy,load_loss_function,calculate_loss
from dataset_preprocessing import return_torch_datasets
from training_utils import process_pct_logic,perform_training_step,print_batch_progress

parser = argparse.ArgumentParser(description='RuleBERT training with PCT')
parser.add_argument('--seed', type=int, default=42, help='Random seed initializer')
parser.add_argument('--include_first', action='store_true', help='Use the first hop rules in PCT')
parser.add_argument('--chain_number', type=int, default=3, help='Depth to train the model')
parser.add_argument('--cuda_number', type=int, default=0, help='CUDA number for GPU training')
parser.add_argument('--apply_PCT', action='store_true', help='Use PCT method during training')
parser.add_argument('--use_ruletext', action='store_true', help='Use rule text during training')
parser.add_argument('--losstype', choices=["wBCE", "SimpleCE", "MSE", "L1loss"], default="wBCE", help="Loss type for training")
parser.add_argument('--model_name_saved', type=str, default="default_name", help='Model save name')
parser.add_argument('--data_dir', type=str, default="data/", help='Data directory')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='PCT hyperparameter')
parser.add_argument('--freeze_layers', action='store_true', help='Freeze layers for faster training')
args = parser.parse_args()

seed_everything(args.seed)

## Fixed parameters
model_arch = 'roberta-large'
max_length = 512
batch_size = 16
eps = 2e-9
epochs = 4
device = torch.device("cuda:"+str(args.cuda_number)) if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG",device)

## Handle the dataset
train_dataloader,val_dataloader,train_dataloader_PD,val_dataloader_PD,rule_lambda = return_torch_datasets(args.data_dir+"chain_rules/Depth_"+str(args.chain_number)+"/",args.include_first,args.use_ruletext,model_arch,max_length,batch_size)
print("size of training dataset and PCT training dataset:",len(train_dataloader),len(train_dataloader_PD))
print("size of validation dataset and PCT validation dataset:",len(val_dataloader),len(val_dataloader_PD))
lambdas=[0 for i in range(len(rule_lambda))]

## Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_arch, num_labels=2)
model = model.to(device)
if args.freeze_layers:
    for name , param in list(model.named_parameters())[:-72]:
        param.requires_grad = False

optimizer = AdamW(model.parameters(),lr=args.lr,eps=eps,weight_decay=0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * len(train_dataloader) * epochs),num_training_steps=int((1 - 0.06) * len(train_dataloader) * epochs))
loss_fct = load_loss_function(args.losstype)
train_dataloader_PD_iter=iter(train_dataloader_PD)

for epoch_i in range(epochs):
    # ========================================
    #               Training
    # ========================================
    args.alpha *= 0.9
    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========\nTraining...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    model.zero_grad()
    for step, batch in enumerate(train_dataloader):
        print_batch_progress(step, train_dataloader, t0)
        loss = perform_training_step(batch, model, device, args, optimizer, scheduler, loss_fct)
        total_train_loss += loss
        if epoch_i > 0 and args.apply_PCT:
            process_pct_logic(model, train_dataloader_PD, train_dataloader_PD_iter, device, args, optimizer,lambdas)
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print(f"\n  Average training loss: {avg_train_loss:.2f}\nTraining epoch took: {training_time}")
    
    model.save_pretrained(f"models/{args.model_name_saved} epoch {epoch_i}")
    # ========================================
    #               Validation
    # ========================================

    print("\nRunning Validation...")

    start_time = time.time()
    model.eval()

    total_accuracy, total_loss, total_conf_accuracy = 0, 0.0, 0
    for batch in val_dataloader:
        b_input_ids, b_input_mask, b_labels, b_weights = [batch[i].to(device) for i in range(4)]
        
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask).logits
            loss = calculate_loss(args.losstype, loss_fct, logits, b_labels, b_weights)
        total_loss += loss.item()
        logits, label_ids = logits.detach().cpu().numpy(), b_labels.to('cpu').numpy()
        total_accuracy += flat_accuracy(logits, label_ids)

    avg_metrics = {metric: total / len(val_dataloader) for metric, total in zip(['Accuracy', 'Loss'], [total_accuracy, total_loss])}
    validation_time = format_time(time.time() - start_time)

    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  Validation took: {validation_time}")