from dataset import Object_Detection_Dataset
from metrics import Metrics
import pandas as pd
import numpy as np
import xmltodict as x2d
from pathlib import Path
import json
import pytesseract as ptr
import cv2 as cv
from PIL import Image, ImageChops
from collections import deque
import matplotlib.pyplot as plt 
from copy import deepcopy
from difflib import SequenceMatcher
import string


class Plate_Detection_Model:
    """
    A class to detect license plates in an image.
    
    Uses both YOLO for license plate bounding boxes, and Tesseract-OCR for license plate characters.
    
    Includes grayscaling, deskewing, color inversion, and...
    lowercase character stripping capabilities.
    """
    def __init__(self, dsize=(416,416), verbose=False, yolover='tiny', bboxdiag=False,\
                 conf=0.3, nms=0.2):
        self.img = None  # The image to work with
        self.img2 = None # The yolo formatted image reshaped for display
        if yolover == 'regular':
            self.net = cv.dnn.readNet("./yolov3/lpr-yolov3.weights",\
                                      "./yolov3/lpr-yolov3.cfg")
        elif yolover == 'tiny':
            self.net = cv.dnn.readNet("./yolov3/lpr-yolov3-tiny.weights",\
                                      "./yolov3/lpr-yolov3-tiny.cfg")
        self.classes = ['license plate']
        self.colors = None
        self.dsize = dsize  # the image size for Yolo network ingestion
        self.osize = None  # the original image size
        self.table = str.maketrans('', '', string.ascii_lowercase)  # a table to strip lowercase
        self.verbose = verbose  # bool for verbose output or simple output
        self.bboxdiag = bboxdiag  # bool, model is for bbox diagnostics only or bbox-OCR pipeline
        self.conf = conf  # NMS confidence threshold
        self.nms = nms  # NMS iou threshold
        
        odd = Object_Detection_Dataset()
        self.Ximg, self.Xcrp, self.Ybox, self.Yplt = odd.get_full_dataset()
        self.mets = Metrics()


    def plot_cv_img(self, input_image):     
        """     
        Converts an image from BGR to RGB and plots
        
        used by:
        get_boxes
        
        parameters:
        input_image (numpy array): the annotated image to plot
        """   
        # change color channels order for matplotlib     
        plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))          

        # For easier view, turn off axis around image     
        plt.axis('off')  
        # Must save prior to show - for show clears the image!
        #plt.savefig("DetectionOutput.jpg")
        plt.show()

        
    def setup_outs(self, img):
        """
        Get the YoloV3 output for an image.
        
        used by:
        procsingle()
        
        parameters:
        img (numpy array): The image to be ingested into YoloV3
        
        returns:
        outs (tuple of numpy array): The YoloV3 output
        """
        layer_name = self.net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        blob = cv.dnn.blobFromImage(image=img,
                                   scalefactor=1/np.max(img),
                                   size=self.dsize)
        #detect objects
        self.net.setInput(blob)
        outs = self.net.forward(output_layer)
        
        return outs
                                    
    
    def get_boxes(self, outs):
        """
        Get the bounding box locations, prediction labels, and prediction confidences
        from the YoloV3 output.
        
        used by:
        procsingle()
        
        parameters:
        outs (tuple of numpy array): The YoloV3 output
        
        returns:
        boxeskp [[int]]: The bbox coordinates to keep after NMS
        """
        class_ids = []
        confidences = []
        boxes = []
        height = self.dsize[0]
        width = self.dsize[1]
        # get the bounding boxes
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0:
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        
        self.img2 = cv.resize(self.img, self.dsize, interpolation=cv.INTER_LANCZOS4)
        implot = deepcopy(self.img2)
        if self.verbose:
            print('Yolo prediction shape: {}'.format(implot.shape))

        objs = []
        confs = []
        boxeskp = []
        font = cv.FONT_HERSHEY_PLAIN
        output = ''
        # get associated information from the bounding boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                objs.append(label)
                output += f'Object: {label} @confidence: {confidences[i]:.2f}, '
                confs.append(confidences[i])
                boxeskp.append(boxes[i])
                color = self.colors[0]
                cv.rectangle(implot, (x, y), (x + w, y + h), color, 2)
                cv.putText(implot, label, (x, y + 30), font, 1, color, 2)
                
        if self.verbose:
            print('confidences: {}'.format(confs))
            print('boxes: {}'.format(boxeskp))
            self.plot_cv_img(implot)
                
        return boxeskp
    
    
    def getSkewAngle(self, src_img):
        """
        Get the skew correction angle for an image of skewed text.

        source:
        https://gist.github.com/zabir-nabil/dfb78f584947ebdb9f29d39c9737b5c6
        
        used by:
        get_plate

        parameters:
        src_img (PIL Image): the license plate crop in original orientation

        returns:
        corang (float): The degrees to correcct skewing
        """
        src_img = np.array(src_img)

        if len(src_img.shape) == 3:
            h, w, _ = src_img.shape
        elif len(src_img.shape) == 2:
            h, w = src_img.shape
        else:
            print('upsupported image type')

        img = cv.medianBlur(src_img, 3)

        edges = cv.Canny(img,  threshold1=30,  threshold2=100, apertureSize=3, L2gradient=True)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=w/4.0, maxLineGap=h/4.0)
        angle = 0.0
        nlines = lines.size

        cnt = 0
        for x1, y1, x2, y2 in lines[0]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if np.abs(ang) <= 30: # excluding extreme rotations
                angle += ang
                cnt += 1
        
        if cnt == 0:
            skewang = 0
        else:
            skewang = (angle / cnt)*180/np.pi
            
        return skewang
    
    
    def rotate_image(self, image, angle):
        """
        Rotate an image

        source:
        https://gist.github.com/zabir-nabil/dfb78f584947ebdb9f29d39c9737b5c6
        
        used by:
        get_plate

        parameters:
        image (PIL Image): The image of text to be rotated
        angle (float): The degrees to rotate the image

        returns:
        result (PIL Image): The rotated image
        """
        image = np.array(image)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        result = Image.fromarray(result)

        return result

    
    def handle_colinv(self, cropim):
        """
        Get the OCR'd plate numbers based on color inversion scheme.
        
        used by:
        get_plate

        parameters:
        cropim (PIL Image): The cropped out license plate from a photo

        returns:
        plate (str): The extracted license plate number
        procim (PIL Image): The processed license plate image
        """
        cropim2 = ImageChops.invert(cropim)
        plate2 = ptr.image_to_string(cropim2).replace(' ', '')
        plate2 = ''.join(filter(str.isalnum, plate2))
        plate2 = plate2.translate(self.table)

        plate1 = ptr.image_to_string(cropim).replace(' ', '')
        plate1 = ''.join(filter(str.isalnum, plate1))
        plate1 = plate1.translate(self.table)

        if len(plate2) > len(plate1):
            plate = plate2
            procim = cropim2
        else:
            plate = plate1
            procim = cropim

        return plate, procim

    
    def get_plate(self, box):
        """
        Manipulate the plate image and perform OCR.
        
        used by:
        procsingle
        
        uses:
        getSkewAngle, rotate_image, handle_colinv
        
        parameters:
        box (list of int): The bbox coordinates
        
        returns:
        plate (str): The OCR'd license plate characters
        """
        pilim = Image.fromarray(self.img).convert('L')
        udscale = self.osize[0] / self.dsize[0]
        lrscale = self.osize[1] / self.dsize[1]

        left = box[0]
        up = box[1]
        right = box[0] + box[2]
        bottom = box[1] + box[3]
        
        left = int(lrscale * left)
        up = int(udscale * up)
        right = int(lrscale * right)
        bottom = int(udscale * bottom)

        cropim = pilim.crop((left, up, right, bottom))
        
        try:
            angle = self.getSkewAngle(cropim)
            cropim = self.rotate_image(cropim, angle)
        except:
            pass
        
        plate, procim = self.handle_colinv(cropim)
        
        if self.verbose:
            procim.show()
            print(plate)
        
        return plate
    
    
    def procsingle(self, image, bboxdiag=False):
        """
        Process a single image and detect objects.
        
        uses:
        setup_outs, get_boxes, get_plate
        
        used by:
        evalbboxes, predict
        
        returns:
        output (str): The string output for the web service
        """
        self.bboxdiag = bboxdiag
        
        self.img = image
        self.osize = (self.img.shape[0], self.img.shape[1])
        
        pilim = Image.fromarray(self.img, 'RGB')
        if self.verbose:
            print('original shape {}'.format(self.osize))
            pilim.show()
        
        outs = self.setup_outs(self.img)
        boxes = self.get_boxes(outs)
        
        if self.bboxdiag == True:
            
            return boxes
        else:
            plates = []
            for box in boxes:
                plate = self.get_plate(box)
                plates.append(plate)
                
            return plates, boxes

        
    def evalbboxes(self):
        """
        Get bbox results from Yolo model on provided vehicle photos.
        
        used by:
        performance_test
        
        uses:
        procsingle
        
        returns:
        boxps (deque of list of int): The predicted bounding boxes
        yboxes (deque of list of int): The provided bounding boxes that correspond
                                        with the predictions
        novboxes (int): The occurences of photos with extraneous boxes
        nodet (int): The occurences of photos lacking a box
        """
        yboxes = deque()
        boxps = deque()
        novboxes = 0
        nodet = 0
        for ximg, ybox in zip(self.Ximg, self.Ybox):
            xscale = ximg.shape[1] / 416
            yscale = ximg.shape[0] / 416
            boxes = self.procsingle(ximg, True)
            for box in boxes:            
                xp1 = (box[0]) * xscale
                xp2 = (box[0] + box[2]) * xscale
                yp1 = (box[1]) * yscale
                yp2 = (box[1] + box[3]) * yscale
                boxp = [xp1, yp1, xp2, yp2]
                
                yboxes.append(ybox)
                boxps.append(boxp)

            if len(boxes) > 1:
                novboxes += 1
            elif len(boxes) == 0:
                nodet += 1

        return boxps, yboxes, novboxes, nodet
    
    
    def evalOCR(self):
        """
        Get plate character results from Tesseract-OCR on provided vehicle photos
        
        used by:
        performance_test
        
        uses:
        getSkewAngle, rotate_image, handle_colinv
        
        returns:
        plateps (deque of str): The OCR'd plate characters
        yplts (deque of str): The truth plate characters
        """
        plateps = deque()
        yplts = deque()
        for xcrp, yplt in zip(self.Xcrp, self.Yplt):
            try:
                xgry = xcrp.convert('L')
                angle = self.getSkewAngle(xgry)
                rotim = self.rotate_image(xgry, angle)
            except:
                pass
            
            plate, procim = self.handle_colinv(rotim)
            
            plateps.append(plate)
            yplts.append(yplt)
            
        return plateps, yplts
        
        
    def performance_test(self):
        """
        Get Yolo bbox and Tesseract-OCR performance metrics.
        
        uses:
        evalbboxes, evalOCR
        
        returns:
        miou (float): mean IOU score
        novboxes (int): # occurences of extraneous bboxes
        nodet (int): # occurences of bboxes lacking
        smmets (float): mean SequenceMatcher score for all plates
        smnbnetmets (float): mean SequenceMatcher score for nonblank plates
        smperfrat (float): proportion of perfect plate matches across all plates
        smnbperfrat (float): proportion of perfect plate matches for nonblank plates
        """
        print('Evaluating Yolo bboxes...')
        boxps, yboxes, novboxes, nodet = self.evalbboxes()
        print('Evaluating Tesseract-OCR...')
        plateps, yplts = self.evalOCR()
        print('Generating performance report...')
        miou, novboxes, nodet, smmets, smnbnetmets, smperfrat, smnbperfrat =\
                self.mets.generate_report(boxps, yboxes, novboxes, nodet, plateps, yplts)
        self.bboxdiag = False
        print('Finished.  Check /results/ folder.')
        
        return miou, novboxes, nodet, smmets, smnbnetmets, smperfrat, smnbperfrat
    
    
    def predict(self, fullarr):
        """
        Run a video through Yolo and Tesseract OCR.
        
        Writes results to file.
        
        uses:
        procsingle
        
        parameters:
        fullarr (np.array): A processed video as an array
        
        returns:
        placecnt (int): number of plates detected
        detplatecnt (int): number of plates with characters detected
        """
        detplates = deque()
        ppth = Path('./results/vidpredictions.txt')

        print('Processing ingested video stream...')
        output = []
        output.append('bboxes and plates detected:\n')
        platecnt = 0
        detplatecnt = 0
        for i in range(0, fullarr.shape[0]):
            plates, boxes = self.procsingle(fullarr[i, :, :, :])
            for plate, box in zip(plates, boxes):
                output.append('{} {}\n'.format(box, plate))
                platecnt += 1
                if plate != '':
                    detplatecnt += 1
                    
        with open(ppth, 'w') as outfile:
            outfile.writelines(output)
        
        strout = '{} nonblank plates detected.  Check /results/ folder for details.'.\
              format(detplatecnt)

        print(strout)

        return platecnt, detplatecnt

