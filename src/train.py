import dataset
import engine
from config import CFG

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, load_metric

import os
import numpy as np

import torch

from sklearn.model_selection import train_test_split



debug = True

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = load_metric('sacrebleu')

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    if not os.path.exists('model_checkpoints/base_model/'):
        model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)
        model.save_pretrained('model_checkpoints/base_model')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained('model_checkpoints/base_model/')


    raw_dataset = load_dataset("europa_eac_tm", language_pair=("pl", "en"))
    X = [i['translation']['pl'] for i in raw_dataset['train']]
    y = [i['translation']['en'] for i in raw_dataset['train']]

    if debug:
        X = X[: CFG.train_batch_size * 8]
        y = y[: CFG.train_batch_size * 8]

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    print(f'train size: {len(x_train)}, valid size: {len(x_valid)}')   

    train_ds = dataset.TranslationDataset(x_train, y_train, tokenizer)
    train_dl = torch.utils.data.DataLoader(train_ds, CFG.train_batch_size, num_workers = 1)

    valid_ds  = dataset.TranslationDataset(x_valid, y_valid, tokenizer)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=CFG.valid_batch_size,  num_workers = 1)

    optimizer = AdamW(model.parameters(), CFG.lr)
    total_steps = CFG.epochs * len(x_train) // CFG.train_batch_size
    warmup_steps = len(x_train) // CFG.train_batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps) 

    best_loss_model_checkpoint = 'model_checkpoints/finetuned_model_best_loss'
    best_bleu_model_checkpoint = 'model_checkpoints/finetuned_model_best_bleu'

    best_loss = float('inf')
    best_bleu = 0

    model.to(device)

    # before finetuning
    all_outputs, all_targets = engine.eval_fn(model, valid_dl, device)
    score = engine.compute_score(all_outputs, all_targets, tokenizer, metrics)
    print(f'bleu before finetuning: {score}')

    for epoch in range(CFG.epochs):
        engine.train_fn(model, optimizer, train_dl, device, scheduler)
        valid_loss = engine.valid_fn(model, valid_dl, device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_pretrained(best_loss_model_checkpoint)

        all_outputs, all_targets = engine.eval_fn(model, valid_dl, device)
        score = engine.compute_score(all_outputs, all_targets, tokenizer, metrics)

        if score < best_bleu:
            best_bleu = score
            model.save_pretrained(best_bleu_model_checkpoint)

        print(f'epoch {epoch + 1}, valid loss: {valid_loss}, bleu score: {score}')