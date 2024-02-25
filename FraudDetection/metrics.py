import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score


class Metrics:
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred
        
        
    def parse_web_output(self):
        """
        Parse the performance data for web output
        
        Returns:
        webout (str): The web output for testing performance
        """
        outdict = classification_report(self.truth, self.pred, output_dict=True)
        f1score = str(outdict['1']['f1-score'])[:5]
        prec = str(outdict['1']['precision'])[:5]
        recl = str(outdict['1']['recall'])[:5]
        spec = str(outdict['0']['recall'])[:5]
        
        rascore = str(roc_auc_score(self.truth, self.pred))[:5]
        
        webout =\
            'f1:{}\nprecision:{} recall1(sensitivity):{} recall0(specificity):{} rocauc:{}'.\
            format(f1score, prec, recl, spec, rascore)
        
        return webout
    
    
    def generate_report(self, rpth):
        """
        Generate a testing performance report.
        
        Metrics include:
        precision, recall, specificity, sensitivity, F1 score, ROC-AUC score
        
        Parameters:
        rpth (pathlib Path): The path to the performance report file
        
        Returns:
        output (str): The performance report contents
        """
        output = 'Model Testing Results:\n\n'
        output += 'note:\n'
        output += 'sensitivity = recall of label 1\n'
        output += 'specificity = recall of label 0\n'
        output += '_____________________________________________________\n'
        output += classification_report(self.truth, self.pred)
        output += '\nROC-AUC score: {}'.format(roc_auc_score(self.truth, self.pred))
        print(output)
        
        with open(rpth, 'w') as outfile:
            outfile.write(output)
        
        return self.parse_web_output()
        