
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import f1_score, fbeta_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class stats:
    def __init__(self, model, X_test, y_test, threshold=0.5):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test


        self.y_predict_proba = model.predict(X_test)
        self.y_predict = np.where(self.y_predict_proba >= 0.5, 1, 0)

        self.default_scores = {'accuracy': accuracy_score(self.y_test, self.y_predict),
                               'f1 score': f1_score(self.y_test, self.y_predict,),
                               'f2 score': fbeta_score(self.y_test, self.y_predict, beta=2),
                               'combined score': (f1_score(self.y_test, self.y_predict) + fbeta_score(self.y_test, self.y_predict, beta=2)) / 2}


    def __str__(self) -> str:
        return f'The model {self.model} has default scores of: \n {self.default_scores}'
        

    def get_threshold_scores(self):

        self.best_threshold = {}
        self.best_f1_score = 0
        self.best_f2_score = 0
        self.best_combined_score = 0
        self.best_accuracy_score = 0

        threshold_scores = {}


        for i in range(1,100, 1):
            
            threshold = i/100

            y_pred = np.where(self.model.predict(self.X_test) >= threshold, 1, 0)

            f1 = f1_score(self.y_test, y_pred)
            f2 = fbeta_score(self.y_test, y_pred, beta=2)
            accuracy = accuracy_score(self.y_test, y_pred)

            combined_score = (f1 + f2) / 2

            threshold_scores[threshold] = (f1, f2, accuracy, combined_score)

            if combined_score > self.best_combined_score:
                self.best_combined_score = combined_score
                self.best_threshold['combined'] = threshold

            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_threshold['f1'] = threshold

            if f2 > self.best_f2_score:
                self.best_f2_score = f2
                self.best_threshold['f2'] = threshold

            if accuracy > self.best_accuracy_score:
                self.best_accuracy_score = accuracy
                self.best_threshold['accuracy'] = threshold

        self.threshold_scores_df = pd.DataFrame.from_dict(threshold_scores, orient='index', columns=['f1', 'f2', 'accuracy', 'combined'])
        
    def plot_threshold_scores(self, ax = None):

        if self.best_threshold is None:
            self.get_threshold_scores()
        
        self.threshold_scores_df.plot(ax=ax)

        plt.xlabel('Threshold')
        plt.ylabel('Score')


    def best_thresholds(self):
        if self.best_threshold is None:
            self.get_threshold_scores()

        return f'''Best F1 score: {self.best_f1_score} at threshold: {self.best_threshold["f1"]} \n
      \n Best F2 score: {self.best_f2_score} at threshold: {self.best_threshold["f2"]}
      \n Best Accuracy score: {self.best_accuracy_score} at threshold: {self.best_threshold["accuracy"]}
      \n Best Combined score: {self.best_combined_score} at threshold: {self.best_threshold["combined"]}'''


    def roc_auc(self, threshold = 0.5):
        self.roc_score = roc_auc_score(self.y_test, np.where(self.y_predict_proba >= threshold, 1, 0))
        print(self.roc_score)


    def plot_roc_auc(self, threshold = 0.5):
        if self.roc_score is None:
            self.roc_auc(threshold)
        
        RocCurveDisplay.from_predictions(self.y_test, np.where(self.y_predict_proba >= threshold, 1, 0))

    
    def plot_precision_recall(self, threshold=0.5):
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, np.where(self.y_predict_proba >= threshold, 1, 0))

        idx = (thresholds >= threshold).argmax()

        plt.plot(thresholds, precision[:-1], label='precision')
        plt.plot(thresholds, recall[:-1], label='recall')
        plt.vlines(threshold, 0, 1.0, colors='r', linestyles='--', label='threshold')

        plt.plot(thresholds[idx], precision[idx], 'bo')
        plt.plot(thresholds[idx], recall[idx], 'o', c='orange')

        plt.axis([0.2, 0.9, 0, 1])

        plt.xlabel('Threshold')
        plt.legend()

        plt.legend()

    def plot_all_best_thresholds(self):
        fig, ax = plt.subplots(4, 3, figsize=(15, 5))
        for i, (k, v) in enumerate(self.best_threshold.items()):
            print(i, k, v)
            threshold = (k, v)

            y_pred = np.where(self.y_predict_proba >= threshold[1], 1, 0)
            
            RocCurveDisplay.from_predictions(self.y_test, y_pred , ax=ax[i][0])


            cm = confusion_matrix(self.y_test, y_pred)
            ConfusionMatrixDisplay(cm).plot(ax=ax[i][1])

            self.plot_threshold_scores(ax=ax[i][2])


    