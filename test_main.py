# main driver for testing model performance on test set
import pandas as pd
import lexicon
import pretrained
import test_eval

def main():
    ## comment all models not being tested ##

    # load test set data from csv
    test_data = pd.read_csv('Data/test.csv')

    # lexicon predictions
    #predictions = lexicon.vader(test_data) # VADER
    #predictions = lexicon.textblob(test_data) # TextBlob

    # pretrained predictions (various models from HuggingFace)
    # models to choose from:
    model1 = 'finiteautomata/bertweet-base-sentiment-analysis'
    model2 = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model3 = 'Seethal/sentiment_analysis_generic_dataset'
    model4 = 'sbcBI/sentiment_analysis_model'
    predictions = pretrained.get_predictions(test_data, model1) # enter chosen model name here

    # pretrained ensemble predictions
    models = [model1, model2, model3, model4] # enter chosen model names here
    mode = 'majority' # enter 'average' or 'majority'
    #predictions  = pretrained.get_ensemble_predictions(test_data, models, mode)

    # evaluate model
    targets = test_data['sentiment']
    test_eval.evaluate(predictions, targets)

if __name__ == '__main__':
    main()