# Evaluate the model against the baseline
import pandas as pd
from sklearn.metrics import classification_report
import sklearn.metrics
import seaborn as sn
import matplotlib.pyplot as plt

def confusion_matrix(predictions, targets):
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ["NEGATIVE", "NEUTRAL", "POSITIVE"]],
                    columns = [i for i in ["NEGATIVE", "NEUTRAL", "POSITIVE"]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate(predictions, targets):
    # print classification report
    print(classification_report(targets, predictions))

    # print confusion matrix
    print(confusion_matrix(predictions, targets))
