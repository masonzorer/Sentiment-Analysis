# Sentiment analysis with pretrained huggingface models
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

# class for tokenizing data for pretrained models
class PretrainedDataset(Dataset):
    def __init__(self, data, model_name):
        self.texts = data['text']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 128
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        encoded_input = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt')
        
        return {    
            'input_ids': encoded_input['input_ids'].flatten(),
            'attention_mask': encoded_input['attention_mask'].flatten(),
        }

# load in model & get predictions
def get_scores(model_name, data):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # create data loader for data
    data['text'] = data['text'].str.slice(0, 128)
    dataset = PretrainedDataset(data, model_name)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # get predictions
    predictions = []
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        logits = logits.detach().cpu()
        scores = softmax(logits)
        predictions.append(scores)
    predictions = np.concatenate(predictions, axis=0)

    return predictions

# get predictions from a single model
def get_predictions(data, model_name):
    predictions = get_scores(model_name, data)
    predictions = np.argmax(predictions, axis=1)
    return predictions

# get predictions from an ensemble of models
def get_ensemble_predictions(data, models, mode):
    
    # get scores from each model passed in
    scores = []
    for model in models:
        scores.append(get_scores(model, data))
    
    if mode == 'average':
        # average scores
        averaged_scores = np.zeros((len(data), 3))
        for i in range(len(data)):
            for j in range(len(models)):
                averaged_scores[i] += scores[j][i]
            averaged_scores[i] /= len(models)

        # get averged method predictions
        predictions = np.argmax(averaged_scores, axis=1)

    elif mode == 'majority':
        # get majority method predictions
        majority_predictions = np.zeros((len(data), 1))
        for i in range(len(data)):
            # get predictions from each model
            votes = []
            for j in range(len(models)):
                votes.append(np.argmax(scores[j][i]))
            majority_predictions[i] = max(set(votes), key=votes.count)

        predictions = majority_predictions

    return predictions





