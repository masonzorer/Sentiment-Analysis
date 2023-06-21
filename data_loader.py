# load in data and create data loader
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def create_data_loader(batch_size, type):
    # load train and development data
    dataset = pd.read_csv('Data/' + type + '.csv')

    # split into X and y
    X = dataset['text']
    y = dataset['sentiment']

    # create data loaders
    dataset = SentimentDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader