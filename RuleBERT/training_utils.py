import time, torch
from utils import format_time, calculate_loss

def print_batch_progress(step, dataloader, start_time):
    """Prints the progress for each batch."""
    if step % (len(dataloader)//5) == 0 and step != 0:
        elapsed = format_time(time.time() - start_time)
        print(f'  Batch {step:5,} of {len(dataloader):5,}. Elapsed: {elapsed}.')

def perform_training_step(batch, model, device, args, optimizer, scheduler, loss_fct):
    """Performs a single training step."""
    b_input_ids, b_input_mask, b_labels, b_weights = [batch[b_num].to(device) for b_num in range(4)]
    output = model(b_input_ids, attention_mask=b_input_mask)
    logits = output.logits
    loss = calculate_loss(args.losstype, loss_fct, logits, b_labels, b_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    model.zero_grad()
    return loss.item()

def process_pct_logic(model, train_dataloader_PD, train_dataloader_PD_iter, device, args, optimizer,lambdas):
    """Processes PCT logic if applicable."""
    try:
        pd_batch_number = next(train_dataloader_PD_iter)
    except StopIteration:
        train_dataloader_PD_iter = iter(train_dataloader_PD)
        pd_batch_number = next(train_dataloader_PD_iter)
    
    b_input_ids,b_input_mask,b_questions_spots,answer_probabilities,b_rule_number = [pd_batch_number[b_num].to(device) for b_num in range(5)]
    o = model(b_input_ids, attention_mask=b_input_mask)
    logits = torch.nn.functional.softmax(o.logits, dim=1)

    PCTloss=torch.Tensor([0]).to(device)
    v=torch.Tensor([1]).to(device)
    for ii in range(len(b_questions_spots)):
        if b_questions_spots[ii]:
            v=torch.abs(logits[ii][0]-v*answer_probabilities[ii])
            PCTloss+=v*lambdas[b_rule_number[ii]]
            lambdas[b_rule_number[ii]]+=args.alpha*v.item()
            v=torch.Tensor([1]).to(device)
        else:
            v=v*logits[ii][1]
    PCTloss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.zero_grad()