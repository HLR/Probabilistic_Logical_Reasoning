import random, os, torch, datetime, torch, json, tqdm, re, numpy as np, pandas as pd
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

def or_proba(rules_probabilities=[], prof=""):
    probas = []
    for k in prof.split("OR"):
        prob_hyp = 100
        for j in [int(i[-1]) - 1 for i in re.findall(r"rule\d", k)]:
            prob_hyp *= rules_probabilities[j] / 100
        probas.append(prob_hyp)
    from problog.program import PrologString
    from problog import get_evaluatable
    prologstring = ""
    for num, i in enumerate(probas):
        prologstring += "a" + str(num) + " . \n"
        prologstring += str(i / 100) + "::b :- a" + str(num) + " . \n"
    prologstring += "query(b). \n"
    p = PrologString(prologstring)
    return list(get_evaluatable().create_from(p).evaluate().items())[0][1]

def convert_to_adverb(x):
    if x > 95:
        return "always", 100
    if x > 85:
        return "usually", 90
    if x > 75:
        return "normally", 80
    if x > 60:
        return "often", 65
    if x > 40:
        return "sometimes", 50
    if x > 20:
        return "occasionally", 30
    if x > 7:
        return "seldom", 15
    return "never", 0

def can_pass(force_answer,asnwer):
    if force_answer==0 and asnwer<=50:
        return True
    if force_answer==1 and asnwer>=50:
        return True
    return False

def create_PCT_dataset(depth_train, data, data_meta, adverb):
    seed_everything(42)
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    model_arch = 'roberta-large'

    with open(data, 'r') as json_file:
        json_list = list(json_file)
    with open(data_meta, 'r') as json_file:
        json_list_meta = list(json_file)

    rule_proba, prev_connect, contexts_PD, questions_PD, probas_PD, is_in_main_PD ,depth_PD = [], [], [], [], [], [], []
    for json_str, meta_str in zip(json_list, json_list_meta):

        main_dataset,meta_dataset = json.loads(json_str),json.loads(meta_str)
        all_main_question_set,all_main_question_dict = set(), dict()
        for i in range(1, 20):
            try:
                meta_dataset['questions']["Q" + str(i)]
            except:
                continue
            c = main_dataset["context"]
            if " not " in meta_dataset['questions']["Q" + str(i)]['question'] or int(meta_dataset['questions']["Q" + str(i)]["QDep"]) > 5:
                continue
            help_proba,prob_hyp = 0,-1
            forced_answer = random.randint(0, 1)
            while (not can_pass(forced_answer,prob_hyp) or prob_hyp == -1):
                prob_hyp = 100
                rules_probabilities = []
                for _ in range(len(meta_dataset['rules'])):
                    prob = max(min(random.gauss(40,60)+help_proba+10*depth_train, 100), 0)
                    prob = convert_to_adverb(int(prob * 100 // 100))[1]
                    rules_probabilities.append(prob)
                if "OR" in meta_dataset['questions']["Q" + str(i)]["proofs"]:
                    prob_hyp = or_proba(rules_probabilities=rules_probabilities, prof=meta_dataset['questions']["Q" + str(i)]["proofs"]) * 100
                else:
                    for j in [int(i_[-1]) - 1 for i_ in re.findall(r"rule\d", meta_dataset['questions']["Q" + str(i)]["proofs"])]:
                        prob_hyp *= rules_probabilities[j] / 100
                help_proba += 5 if forced_answer and prob_hyp < 50 else -5 if not forced_answer and prob_hyp > 50 else 0
                if int(meta_dataset['questions']["Q" + str(i)]["QDep"]) == 0:
                    break
            if " not " in meta_dataset['questions']["Q" + str(i)]['question']:
                prob_hyp = 100 - prob_hyp
            c = main_dataset["context"]
            for num, j in enumerate(meta_dataset['rules'].keys()):
                adverb_converted = convert_to_adverb(int(rules_probabilities[num]))[0]
                if not adverb:
                    c = c.replace(meta_dataset['rules'][j]["text"],"With a probability of " + str(int(rules_probabilities[num])) + ".00 percent , " + meta_dataset['rules'][j]["text"])
                else:
                    c = c.replace(meta_dataset['rules'][j]["text"], adverb_converted + " , " + meta_dataset['rules'][j]["text"])
            
            all_main_question_set.add(meta_dataset['questions']["Q" + str(i)]['question'])
            all_main_question_dict[ meta_dataset['questions']["Q" + str(i)]['question'] ] = (c,rules_probabilities,meta_dataset['questions']["Q" + str(i)]["QDep"])
        
        all_proofs = meta_dataset['allProofs']
        probs_dict = dict()
        cur_index = -1
        for i in range(1, len(all_proofs.split("@"))):
            if cur_index == 15:
                break
            for j in all_proofs.split("@")[i][2:].split("]")[:-1]:
                if "OR" in j.split(".")[1]:
                    continue
                r_s = [int(i[-1]) - 1 for i in re.findall(r"rule\d", j.split(".")[1])]
                if len(r_s) < 1:
                    continue
                if cur_index == 15:
                    break
                cur_index += 1
                rule_missing = False
                pd_question_j=j.split(".")[0][1:]+"."
                if pd_question_j in all_main_question_dict:
                    c = all_main_question_dict[pd_question_j][0]
                    rules_probabilities=all_main_question_dict[pd_question_j][1]
                else:
                    help_proba,prob_hyp = 0, -1
                    forced_answer = random.randint(0, 1)
                    while (not can_pass(forced_answer,prob_hyp) or prob_hyp == -1):
                        prob_hyp = 100
                        rules_probabilities = []
                        for _ in range(len(meta_dataset['rules'])):
                            prob = max(min(random.gauss(40,60)+help_proba+10*depth_train, 100), 0)
                            prob = convert_to_adverb(int(prob * 100 // 100))[1]
                            rules_probabilities.append(prob)
                        for j_ in r_s:
                            prob_hyp *= rules_probabilities[j_] / 100
                        help_proba += 5 if forced_answer and prob_hyp < 50 else -5 if not forced_answer and prob_hyp > 50 else 0

                    c = main_dataset["context"]
                    for num, _j_ in enumerate(meta_dataset['rules'].keys()):
                        adverb_converted, _ = convert_to_adverb(int(rules_probabilities[num]))
                        if not adverb:
                            c = c.replace(meta_dataset['rules'][_j_]["text"], "With a probability of " + str(int(rules_probabilities[num])) + ".00 percent , " +meta_dataset['rules'][_j_]["text"])
                        else:
                            c = c.replace(meta_dataset['rules'][_j_]["text"], adverb_converted + " , " + meta_dataset['rules'][_j_]["text"])

                prob_hyp = 100
                for j_ in r_s:
                    try:
                        prob_hyp *= rules_probabilities[j_] / 100
                    except:
                        rule_missing = True
                if rule_missing:
                    continue
                if r_s:
                    rule_proba.append(rules_probabilities[r_s[-1]])
                else:
                    rule_proba.append(-1)

                probs_dict[j.split(".")[1].strip("[()")] = cur_index
                index = j.split(".")[1].rfind("rule", 0, j.split(".")[1].rfind("rule") - 1)
                if index == -1:
                    prev_connect.append(-1)
                else:
                    index += 6
                    final_name = j.split(".")[1][0:index] + ")"
                    if final_name.strip("[()") in probs_dict:
                        prev_connect.append(probs_dict[final_name.strip("[()")])
                    else:
                        prev_connect.append(-1)
                contexts_PD.append(c)
                questions_PD.append(j.split(".")[0][1:]+".")
                is_in_main_PD.append(j.split(".")[0][1:]+"." in all_main_question_set)
                probas_PD.append(prob_hyp)
                if pd_question_j in all_main_question_dict:
                    depth_PD.append(all_main_question_dict[pd_question_j][2])
                else:
                    depth_PD.append(-1)

        if not cur_index == -1:
            for _ in range(cur_index + 1, 16):
                contexts_PD.append(contexts_PD[-1])
                questions_PD.append(questions_PD[-1])
                probas_PD.append(probas_PD[-1])
                prev_connect.append(-1)
                rule_proba.append(-1)
                is_in_main_PD.append(False)
                depth_PD.append(-1)

    contexts_PD_batch_4, questions_PD_batch_4, probas_PD_batch_4, prev_connect_batch_4, rule_proba_batch_4, is_in_main_PD_batch_4,depth_PD_batch_4 = [], [], [], [], [], [],[] ##
    if not depth_train == 0:
        even = True
        for i in range(0, len(contexts_PD), 16):
            for j in range(0, 16):
                cur_node = 15 - j + i
                if not prev_connect[cur_node] == -1:

                    prev_node = i + prev_connect[cur_node]

                    contexts_PD_batch_4.append(contexts_PD[prev_node])
                    questions_PD_batch_4.append(questions_PD[prev_node])
                    probas_PD_batch_4.append(probas_PD[prev_node])
                    prev_connect_batch_4.append(-1)
                    rule_proba_batch_4.append(rule_proba[prev_node])
                    is_in_main_PD_batch_4.append(is_in_main_PD[prev_node])##
                    depth_PD_batch_4.append(depth_PD[prev_node])

                    contexts_PD_batch_4.append(contexts_PD[cur_node])
                    questions_PD_batch_4.append(questions_PD[cur_node])
                    probas_PD_batch_4.append(probas_PD[cur_node])
                    if even:
                        prev_connect_batch_4.append(0)
                    else:
                        prev_connect_batch_4.append(2)
                    even = not even
                    rule_proba_batch_4.append(rule_proba[cur_node])
                    is_in_main_PD_batch_4.append(is_in_main_PD[cur_node])
                    depth_PD_batch_4.append(depth_PD[cur_node])

        if not len(contexts_PD_batch_4) % 4 == 0:
            contexts_PD_batch_4.pop()
            contexts_PD_batch_4.pop()

            questions_PD_batch_4.pop()
            questions_PD_batch_4.pop()

            probas_PD_batch_4.pop()
            probas_PD_batch_4.pop()

            prev_connect_batch_4.pop()
            prev_connect_batch_4.pop()

            rule_proba_batch_4.pop()
            rule_proba_batch_4.pop()

            is_in_main_PD_batch_4.pop()
            is_in_main_PD_batch_4.pop()
            
            depth_PD_batch_4.pop()
            depth_PD_batch_4.pop()
    contexts_PD, questions_PD, probas_PD, prev_connect, rule_proba, is_in_main_PD, depth_PD = contexts_PD_batch_4 \
        , questions_PD_batch_4, probas_PD_batch_4, prev_connect_batch_4, rule_proba_batch_4 ,is_in_main_PD_batch_4,depth_PD_batch_4 
    
    tokenizer = AutoTokenizer.from_pretrained(model_arch)
    train_input_ids_PD = []
    train_attention_masks_PD = []
    for c, h in tqdm.tqdm(zip(contexts_PD, questions_PD)):
        encoded = tokenizer.encode_plus(c, h, max_length=384, truncation=True, return_tensors='pt',
                                        padding='max_length')

        train_input_ids_PD.append(encoded['input_ids'])
        train_attention_masks_PD.append(encoded['attention_mask'])
    try:
        train_input_ids_PD = torch.cat(train_input_ids_PD, dim=0)
        train_attention_masks_PD = torch.cat(train_attention_masks_PD, dim=0)
    except:
        train_input_ids_PD = torch.LongTensor(train_input_ids_PD)
        train_attention_masks_PD = torch.LongTensor(train_attention_masks_PD)

    train_proba_PD, prev_connect, rule_proba, is_in_main_PD, depth_PD,  = torch.FloatTensor(probas_PD) , torch.LongTensor(prev_connect), torch.FloatTensor(rule_proba),torch.LongTensor(is_in_main_PD),torch.LongTensor(depth_PD)
    train_dataset_PD = TensorDataset(train_input_ids_PD, train_attention_masks_PD, train_proba_PD, prev_connect, rule_proba,is_in_main_PD,depth_PD)
    train_dataloader_PD = DataLoader(dataset=train_dataset_PD, sampler=SequentialSampler(train_dataset_PD), batch_size=4)
    return train_dataloader_PD