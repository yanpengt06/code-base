"""
    train/validate loop
"""

import numpy as np, argparse, time, pickle, random
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from tqdm import tqdm
import json


def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False,
                        scheduler=None):
    losses, preds, labels = [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
        # dataloader = tqdm(dataloader)
        # print(f"current roberta learning rate is :{optimizer.param_groups[0]['lr']}")        
        # print(f"current other learning rate is :{optimizer.param_groups[1]['lr']}")                
    else:
        model.eval()

    cnt = 0
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        input_ids, att_mask, token_type_ids, label, _ = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            input_ids = input_ids.cuda()
            att_mask = att_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            label = label.cuda()

        # print(speakers)
        output = model(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_type_ids)  # (B, T, C)
        logits = output.logits  # B x C
        loss = loss_function(logits, label)  # B x C --- B
        label = label.cpu().numpy().tolist()
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()  # (B x T, C)
        preds += pred
        labels += label
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            # print(f"current learning rate is :{optimizer.param_groups[0]['lr']} and {optimizer.param_groups[1]['lr']}")                          
            optimizer.step()
            scheduler.step()

    # if train:
    #     # scheduler.step()
    #     print(f"current learning rate is :{optimizer.param_groups[0]['lr']} and {optimizer.param_groups[1]['lr']}")        

    if preds != []:
        new_preds = []
        new_labels = []
        # for i,label in enumerate(labels):
        #     for j,l in enumerate(label):
        #         if l != -1:
        #             new_labels.append(l)
        #             new_preds.append(preds[i][j])
        for i, label in enumerate(labels):
            if label != -1:
                new_labels.append(label)
                new_preds.append(preds[i])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # if not train:
    #     print(classification_report(new_labels, new_preds))
    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP', 'jddc', 'sst2']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore
    else:  # DailyDialog
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(0, 3))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_micro_fscore, avg_macro_fscore


def save_badcase(model, dataloader, cuda, args, speaker_vocab, label_vocab):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []

    model.eval()

    for data in dataloader:

        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        features, label, adj, s_mask, s_mask_onehot, lengths, speaker, utterances = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()

        # print(speakers)
        log_prob = model(features, adj, s_mask, s_mask_onehot, lengths)  # (B, N, C)

        label = label.cpu().numpy().tolist()  # (B, N)
        pred = torch.argmax(log_prob, dim=2).cpu().numpy().tolist()  # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker

        # finished here

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return

    cases = []
    for i, d in enumerate(dialogs):
        case = []
        for j, u in enumerate(d):
            case.append({
                'text': u,
                'speaker': speaker_vocab['itos'][speakers[i][j]],
                'label': label_vocab['itos'][labels[i][j]] if labels[i][j] != -1 else 'none',
                'pred': label_vocab['itos'][preds[i][j]]
            })
        cases.append(case)

    with open('badcase/%s.json' % (args.dataset_name), 'w', encoding='utf-8') as f:
        json.dump(cases, f)

    # print(preds.tolist())
    # print(labels.tolist())
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        print('badcase saved')
        print('test_f1', avg_fscore)
        return
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        print('badcase saved')
        print('test_micro_f1', avg_micro_fscore)
        print('test_macro_f1', avg_macro_fscore)
        return
