# import pandas as pd
import numpy as np
# from dataset import Object_Detection_Dataset
# from model import Plate_Detection_Model
from pathlib import Path
from difflib import SequenceMatcher
from collections import deque
# from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score


class Metrics:
    def __init__(self, rpthstr='./results/testing_provided_photos.txt'):
        self.rpth = Path(rpthstr)
        

    def get_iou(self, boxA, boxB):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - boxA: The first bounding box as a list of coordinates [x1, y1, x2, y2].
        - boxB: The second bounding box as a list of coordinates [x1, y1, x2, y2].

        Returns:
        - The IoU as a float.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
    
    
    def generate_report(self, boxps, yboxes, novboxes, nodet, plateps, yplts):
        """
        Generate a testing performance report.
        
        Metrics include:
        precision, recall, specificity, sensitivity, F1 score, ROC-AUC score
        
        Parameters:
        rpth (pathlib Path): The path to the performance report file
        
        Returns:
        miou (float): mean IOU score
        novboxes (int): # occurences of extraneous bboxes
        nodet (int): # occurences of bboxes lacking
        smmets (float): mean SequenceMatcher score for all plates
        smnbnetmets (float): mean SequenceMatcher score for nonblank plates
        smperfrat (float): proportion of perfect plate matches across all plates
        smnbperfrat (float): proportion of perfect plate matches for nonblank plates
        """
        # First get IOU metrics for Yolo evaluation
        ious = deque()
        for boxp, ybox in zip(boxps, yboxes):
            iou = self.get_iou(ybox, boxp)
            ious.append(iou)
            
        miou = np.mean(ious)
        output = []
        output.append('Yolo bbox testing results:\n')
        output.append('mean IOU: {}\n'.format(miou))
        output.append('images with extraneous boxes: {}\n'.format(novboxes))
        output.append('images lacking boxes: {}\n'.format(nodet))
        output.append('')
        
        # Now get SequenceMatcher metrics for Tesseract-OCR evaluation
        smmets = []  # sequencematcher metrics
        smnbnetmets = []  # non-blank net metrics
        smnbperfcnt = 0 # non-blank perfect LCS count
        smnbcnt = 0 # non-blank count
        smcnt = 0  # all plates count
        for platep, yplt in zip(plateps, yplts):
            smmet = SequenceMatcher(None, yplt, platep).ratio()
            smcnt += 1
            smmets.append(smmet)
            if platep != '':
                smnbnetmets.append(smmet)
                smnbcnt += 1
                if smmet == 1:
                    smnbperfcnt += 1

        smnbperfrat = smnbperfcnt / smnbcnt
        smperfrat = smnbperfcnt / smcnt
        smmets = np.mean(smmets)
        smnbnetmets = np.mean(smnbnetmets)
        
        output.append('\nTesseract-OCR testing results:\n')
        output.append('LCS mean, all plates: {}\n'.format(smmets))
        output.append('LCS mean, only if chars detected: {}\n'.format(smnbnetmets))
        output.append('proportion perfect LCS, all plates: {}\n'.format(smperfrat))
        output.append('proportion perfect LCS, only if chars detected: {}\n'.format(smnbperfrat))
        
        with open(self.rpth, 'w') as outfile:
            outfile.writelines(output)
            
        return miou, novboxes, nodet, smmets, smnbnetmets, smperfrat, smnbperfrat


