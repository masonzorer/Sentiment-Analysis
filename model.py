# training model
import torch
import transformers

# define model class
class SentimentClassifier(torch.nn.Module):
    def __init__(self, dropout):
        super(SentimentClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(768, 3)
    
    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.dropout(output)
        return self.out(output)

def train(train_loader, dev_loader, epochs, learning_rate, dropout):
    
    # initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    update = 0
    # training loop
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            print(f'Update: {update}, Loss:  {loss.item()}')
            update += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate model multiple times per epoch
            if update % 100 == 0:
                model.eval()
                correct_predictions = 0
                n_examples = 0
                with torch.no_grad():
                    for i, batch in enumerate(dev_loader):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                        _, preds = torch.max(outputs, dim=1)
                        
                        correct_predictions += torch.sum(preds == labels)
                        n_examples += len(labels)
                    
                    accuracy = correct_predictions.double() / n_examples
                    print(f'Accuracy: {accuracy.item()}')
                model.train()
        
        print(f'Start of Epoch: {epoch+2}')


