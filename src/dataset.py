from config import CFG

class TranslationDataset:
    def __init__(self, x, y, tokenizer, max_len=None):
        self.x = x # sentence in polish
        self.y = y # sentence in english
        self.tokenizer = tokenizer 
        self.max_len = max_len
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sentence = self.x[idx]
        tokenized_sentence = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True)

        if self.y is not None:
            translation = self.y[idx]
            tokenized_translation = self.tokenizer(translation, return_tensors='pt', padding='max_length', truncation=True)
            tokenized_sentence['targets'] = tokenized_translation['input_ids']
        
        return tokenized_sentence

if __name__ == '__main__':
    # tests

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from config import CFG

    raw_dataset = load_dataset(CFG.dataset_name, language_pair=("pl", "en"))
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    x = [i['translation']['pl'] for i in raw_dataset['train']]
    y = [i['translation']['en'] for i in raw_dataset['train']]

    dataset = TranslationDataset(
        x = x,
        y = y,
        tokenizer = tokenizer,
    )

    print(dataset[0])


