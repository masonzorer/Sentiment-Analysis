# load and preprocess training data
import pandas as pd

def load_data():
    # load dataset csv file
    dataset = pd.read_csv('Data/All_Data.csv')

    # print size of dataset
    print('Dataset size: ', dataset.shape)

    # shuffle dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # split dataset into training, dev, and testing sets
    train = dataset.sample(frac=0.8, random_state=200)
    test = dataset.drop(train.index)
    dev = test.sample(frac=0.5, random_state=200)
    test = test.drop(dev.index)

    # print size of each set
    print('Training set size: ', train.shape)
    print('Dev set size: ', dev.shape)
    print('Testing set size: ', test.shape)

    # save each set to csv file
    train.to_csv('Data/train.csv', index=False)
    dev.to_csv('Data/dev.csv', index=False)
    test.to_csv('Data/test.csv', index=False)

def main():
    load_data()

if __name__ == '__main__':
    main()