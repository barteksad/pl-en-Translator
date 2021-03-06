import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

import datasets

def loss_fn(y_pred, y_true):
    loss = cross_entropy(y_pred.permute(0,2,1), y_true)
    return loss

def train_fn(model, optimizer, dataloader, device, scheduler=None):
    model.train()

    running_loss = 0
    num_steps = 0
    train_loss = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for bi, b in pbar:
        num_steps += 1

        input_ids = b['input_ids'].to(device)
        att_mask = b['attention_mask'].to(device)
        targets = b['targets'].to(device)

        optimizer.zero_grad()
        preds = model(
            input_ids = input_ids,
            attention_mask = att_mask,
            decoder_input_ids = targets
        )

        loss = loss_fn(preds.logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        train_loss = running_loss / num_steps
        pbar.set_description(f'train loss: {train_loss:.4f}')

    return train_loss

def valid_fn(model, dataloader, device):
    model.eval()
    all_valid_loss = []

    with torch.no_grad():
        for b in tqdm(dataloader, total=len(dataloader)):
            input_ids = b['input_ids'].to(device)
            att_mask = b['attention_mask'].to(device)
            targets = b['targets'].to(device)

            preds = model(
                input_ids = input_ids,
                attention_mask = att_mask,
                decoder_input_ids = targets
            )

            all_valid_loss.append(loss_fn(preds.logits, targets).item())

        return np.mean(all_valid_loss)

def eval_fn(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for b in tqdm(dataloader, total=len(dataloader)):
            input_ids = b['input_ids'].to(device)
            att_mask = b['attention_mask'].to(device)
            targets = b['targets']

            preds = model.generate(
                input_ids = input_ids,
                attention_mask = att_mask,
            )

            all_outputs.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        
        return all_outputs, all_targets


def compute_score(all_outputs, all_targets, tokenizer, metrics : datasets.Metric):
    all_outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
    all_targets = tokenizer.batch_decode(all_targets, skip_special_tokens=True)
    all_targets = np.expand_dims(all_targets, axis=1)
    
    score = metrics.compute(predictions=all_outputs, references = all_targets)

    return score['score']
