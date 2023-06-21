# load data and train model
import pandas as pd
import data_loader
import model

def train_model():
    
    # Hyperparameters
    batch_size = 4
    epochs = 3
    learning_rate = 2e-5
    dropout_rate = 0.3

    # load data
    train_loader = data_loader.create_data_loader(batch_size, 'train')
    dev_loader = data_loader.create_data_loader(batch_size, 'dev')

    # train model
    model.train(train_loader, dev_loader, epochs, learning_rate, dropout_rate)


def main():
    train_model()

if __name__ == '__main__':
    main()