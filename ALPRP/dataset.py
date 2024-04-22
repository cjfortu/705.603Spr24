import numpy as np
import json
from pathlib import Path
from collections import deque
from PIL import Image
from sklearn.model_selection import train_test_split, KFold


class Object_Detection_Dataset:
    """
    A class to partition the dataset.
    """
    def __init__(self, cocopthstr='./COCO_YOLO_dat/COCO_annot.json', dsplit=(0.8, 0.1, 0.1)):
        """
        Ingest the dataset and make initial train/val/test splits.
        
        There are two sets of X and Y.  
        
        One for vehicle images and bounding boxes, for Yolo metrics.
        
        The other for plate images and plate characters, for Tesseract-OCR metrics
        
        Parameters:
        cocopthstr (str): The COCO json file path
        split (tuple of float): The train/val/test proportions
        """
        self.dsplit = dsplit
        
        # extract from COCO json file
        cocopth = Path(cocopthstr)
        with open(cocopth) as jsonfile:
            cocodat = json.load(jsonfile)
        
        self.Ximg = []  # full vehicle images
        self.Xcrp = []  # cropped vehicle images according to provided boxes
        self.Ybox = []  # provided box coordinates
        self.Yplt = []  # license plate character truth
        # iterate through COCO json file
        for annotation in cocodat['annotations']:
            image = Image.open('./images/{}.jpeg'.format(annotation['image_id']))
            image = np.array(image)
            truth = annotation['content']
            
            left = annotation['bbox'][0]
            up = annotation['bbox'][1]
            right = annotation['bbox'][0] + annotation['bbox'][2]
            bottom = annotation['bbox'][1] + annotation['bbox'][3]
            boxt = [left, up, right, bottom]
            cropim = Image.fromarray(image).crop(list(boxt))
            
            self.Ximg.append(image)
            self.Xcrp.append(cropim)
            self.Ybox.append(boxt)
            self.Yplt.append(truth)
        
        self.Ximg_tv, self.Ximg_test, self.Ybox_tv, self.Ybox_test =\
                train_test_split(self.Ximg, self.Ybox, test_size=self.dsplit[2])
        self.Ximg_train, self.Ximg_val, self.Ybox_train, self.Ybox_val =\
                train_test_split(self.Ximg_tv, self.Ybox_tv,\
                test_size = self.dsplit[1] / (self.dsplit[0] + self.dsplit[2]))
        
        self.Xcrp_tv, self.Xcrp_test, self.Yplt_tv, self.Yplt_test =\
                train_test_split(self.Xcrp, self.Yplt, test_size=self.dsplit[2])
        self.Xcrp_train, self.Xcrp_val, self.Yplt_train, self.Yplt_val =\
                train_test_split(self.Xcrp_tv, self.Yplt_tv,\
                test_size = self.dsplit[1] / (self.dsplit[0] + self.dsplit[2]))
    
    
    def get_full_dataset(self):
        """
        Get the full dataset (without splitting)
        """
        return (self.Ximg, self.Xcrp, self.Ybox, self.Yplt)
    
    
    def get_training_dataset(self):
        """
        Get the training data for the 1-fold CV case
        """
        return (self.Ximg_train, self.Xcrp_train, self.Ybox_train, self.Yplt_train)
        
        
    def get_validation_dataset(self):
        """
        Get the validation data for the 1-fold CV case
        """
        return (self.Ximg_val, self.Xcrp_val, self.Ybox_val, self.Yplt_val)
    
    
    def get_testing_dataset(self):
        """
        Get the testing data
        """
        return (self.Ximg_test, self.Xcrp_test, self.Ybox_test, self.Yplt_test)
    
    
    def get_noval_training_dataset(self):
        """
        Get the training data for the no CV case
        """
        return (self.Ximg_tv, self.Xcrp_tv, self.Ybox_tv, self.Yplt_tv)
    
    
    def get_nfoldCV_datasets(self):
        """
        Split the non-test data into n-fold sets for cross validation.
        
        Parameters:
        n_splits (int): The desired number of splits
        
        Returns:
        X_trains (list of np.array): The X training datasets
        X_vals (list of np.array): The X validation datasets
        Y_trains (list of np.array): The Y training datasets
        Y_vals (list of np.array): The Y validation datasets
        """
        n_splits = int(self.dsplit[0] / self.dsplit[1]) + 1
        Ximg_trains = []
        Ybox_trains = []
        Ximg_vals = []
        Ybox_vals = []
        Xcrp_trains = []
        Yplt_trains = []
        Xcrp_vals = []
        Yplt_vals = []
        
        skf = KFold(n_splits=n_splits, shuffle=True)
        for t_idx, v_idx in skf.split(self.Ximg_tv, self.Ybox_tv):
            Ximg_trains.append(self.Ximg_tv[t_idx])
            Ybox_trains.append(self.Ybox_tv[t_idx])
            Ximg_vals.append(self.Ximg_tv[v_idx])
            Ybox_vals.append(self.Ybox_tv[v_idx])
            
        for t_idx, v_idx in skf.split(self.Xcrp_tv, self.Yplt_tv):
            Xcrp_trains.append(self.Xcrp_tv[t_idx])
            Yplt_trains.append(self.Yplt_tv[t_idx])
            Xcrp_vals.append(self.Xcrp_tv[v_idx])
            Yplt_vals.append(self.Yplt_tv[v_idx])
            
        return Ximg_trains, Ximg_vals, Ybox_trains, Ybox_vals,\
                Xcrp_trains, Xcrp_vals, Yplt_trains, Yplt_vals

    
# if __name__ == "__main__":
#     oddataset = Object_Detection_Dataset('./COCO_YOLO_dat/COCO_annot.json', (0.8, 0.1, 0.1))
#     X_trains, X_vals, Y_trains, Y_vals = oddataset.get_nfoldCV_datasets()
#     print(len(X_trains))
#     print(len(X_vals))
#     print(len(Y_trains))
#     print(len(Y_vals))