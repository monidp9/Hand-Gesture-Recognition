import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class NNClassifier():
    def __init__(self):
        self.model = tf.keras.models.load_model("nn_model/nn_classifier.hdf5")

    def predict(self, X):
        X = np.array(X)

        result = self.model.predict(X[np.newaxis, :], verbose=0)
        result = np.squeeze(result)
        return np.argmax(result)

    def evaluate(self, X_test, y_test, matrix=True):
        Y_pred = self.model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)

        print('Classification Report')
        print(classification_report(y_test, y_pred))

        if matrix:
            labels = sorted(list(set(y_test)))
            cmx_data = confusion_matrix(y_test, y_pred, labels=labels)
            
            labels = ['OpenPalm', 'Victory', 'ClosedFist', 'PoitingUp', 'NoGesture']
            df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
        
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
            ax.set_ylim(len(set(y_test)), 0)
            plt.show()
        

if __name__ == '__main__':
    # Run the current file in order to evaluate the neural network classifier

    RANDOM_SEED = 142
    dataset_path = './nn_model/keypoint.csv'

    X_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0))

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    nn_classifier = NNClassifier()
    nn_classifier.evaluate(X_test, y_test)
