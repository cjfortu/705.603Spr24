# import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from sklearn.metrics import accuracy_score


class Metrics:
    def __init__(self, mode, Y_vt, y_preds, rpthstr=r'./results/output.txt'):
        self.mode = mode
        self.Y_vt = Y_vt
        self.y_preds = y_preds
        self.rpth = Path(rpthstr)
        self.proximalperf = None
        self.accuracy = None
        

    def get_proximalperf(self):
        """
        Compute proximalperf.

        Assigns a value of 1 for each correct label, and a value (5-abs(diff))/nclasses
        for each incorrect label. For example, if *y_truth=5* and *y_pred=2*,
        then the value will be 2/5.

        This allows some lenience if scores were only a single star rating off.
        This is important because there is inherent subjectivity and error in the
        star ratings versus the text.
        """
        truidxs = np.argwhere(self.Y_vt > 0)[:, 1]
        prdidxs = np.argmax(self.y_preds, axis=1)
        tot = (5 - np.abs(truidxs - prdidxs)) / self.y_preds.shape[1]
        self.proximalperf = np.sum(tot)
        self.proximalperf /= self.y_preds.shape[0]
    
    
    def get_accuracy(self):
        """
        Compute the accuracy score.
        """
        truth = np.argmax(self.Y_vt, axis=1)
        preds = np.argmax(self.y_preds, axis=1)
        self.accuracy = accuracy_score(truth, preds)
    
    
    def generate_report(self):
        """
        Generate a testing performance report.
        
        Returns:
        y_preds (int): The predicted star rating
        accuracy (float): The accuracy
        proximalperf (float): The proximalperf score
        """
        output = []
        output.append('\n{} entry testing results:\n'.format(self.mode))
        if self.mode == 'single':
            output.append('{}'.format(self.y_preds))
        elif self.mode == 'batch':
            self.get_proximalperf()
            self.get_accuracy()
            output.append('accuracy: {}\n'.format(self.accuracy))
            output.append('proximalperf: {}\n'.format(self.proximalperf))
        
        with open(self.rpth, 'a') as outfile:
            outfile.writelines(output)
        
        if self.mode == 'single':
            return self.y_preds
        elif self.mode == 'batch':
            return self.accuracy, self.proximalperf


