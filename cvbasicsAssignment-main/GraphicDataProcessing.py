import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt 
from copy import deepcopy


class ObjectDetection:
    """
    A class to detect objects in an image, and to stress-test the detection ability after image manipulation.
    """
    def __init__(self):
        self.img = None  # The image to work with
        self.net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")  # the CNN
        with open("coco.names", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = None
        self.dsize = (416, 416)

        
    def setup_outs(self, img):
        """
        Get the YoloV3 output for an image.
        
        used by:
        procsingle(), shrink(), rotate(), noise()
        
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
                                    
    
    def pred(self, outs):
        """
        Get the bounding box locations, prediction labels, and prediction confidences
        from the YoloV3 output.
        
        used by:
        procsingle(), shrink(), rotate(), noise()
        
        parameters:
        outs (tuple of numpy array): The YoloV3 output
        
        returns:
        objs (list of str): The detected object labels
        confs (list of float): The detection confidences
        output (str): The string output for the web service
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
                if confidence > 0.5:
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

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        objs = []
        confs = []
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
                color = self.colors[i]
                cv.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                cv.putText(self.img, label, (x, y + 30), font, 3, color, 3)
                
        return objs, confs, output
    
    
    def procsingle(self):
        """
        Process a single image and detect objects.
        
        uses:
        setup_outs(), pred()
        
        returns:
        output (str): The string output for the web service
        """
        self.img = cv.imread('./Pictures/recimg')
        outs = self.setup_outs(self.img)
        _, _, output = self.pred(outs)
        
        return output
                
        
    def plotdeg(self, allobjs, allconfs, params, meas, activity):
        """
        Plot the prediction confidence vs image manipulation, and identify
        points of prediction loss.
        
        used by:
        manipulate()
        
        parameters:
        allobjs (list of list of str): The objects identified at each manipulation parameter value
        allconfs (list of list of float): The prediction confidence at each manipulation parameter value
        params (list of float): The manipulation parameter values
        meas (str): The parameter being manipulated
        activity (str): The description of the manipulation activity
        """
        maxlen = np.max([len(allobj) for allobj in allobjs])
        objsac = [[] for i in range(0, maxlen)]  # each object at each parameter value
        confsac = deepcopy(objsac)  # each confidence level at each parameter value
        paramsac = deepcopy(objsac)  # the parameter values with a prediction
        paramsls = [deepcopy(params) for i in range(0, maxlen)]  # parameter values with prediction lost
        # create a list for each object and its confidence at each parameter value
        for objs, confs, param in zip(allobjs, allconfs, params):
            for i, (obj, conf) in enumerate(zip(objs, confs)):
                objsac[i].append(obj)
                confsac[i].append(conf)
                paramsac[i].append(param)
                paramsls[i].remove(param)  # also record the parameters at which prediction was lost
        
        # For each object detected, plot the prediction confidences (or lack of) across manipulation params
        for i, (objac, confac, paramac, paramls) in\
                enumerate(zip(objsac, confsac, paramsac, paramsls)):
            label = '{}_{}'.format(objac[0], i)
            print('{} lost at {}: {}'.format(label, meas, paramls))
            plt.scatter(paramac,
                     confac,
                     label = label)
            
        plt.xlabel(meas)
        plt.xticks(params, rotation='vertical')
        plt.ylabel('prediction confidence')
        plt.title('Prediction Confidence Degradation According to {}'.\
                  format(activity))
        if activity == 'Size Reduction':
            plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True, linewidth=1, linestyle='dashdot')
        plt.show()
        
        
    def shrink(self):
        """
        Shrink the object according to a scaling factor
        
        uses:
        setup_outs(), pred()
        
        used by:
        manipulate()
        
        returns:
        allobjs (list of list of str): The objects identified at each manipulation parameter value
        allconfs (list of list of float): The prediction confidence at each manipulation parameter value
        params (list of float): The manipulation parameter values
        """
        height = self.img.shape[0]
        width = self.img.shape[1]
        allobjs = []
        allconfs = []
        params = []
        for i in reversed(range(1, 21, 1)):
            param = i / 100
            ht = int(height * param)
            wd = int(width * param)
            nimg = cv.resize(self.img,
                             dsize=(ht, wd),
                             interpolation=cv.INTER_LANCZOS4)
            outs = self.setup_outs(nimg)
            objs, confs, _ = self.pred(outs)
            allobjs.append(objs)
            allconfs.append(confs)
            params.append(param)
        
        return allobjs, allconfs, params
        
    
    def rotate(self):
        """
        Rotate the object by degrees
        
        uses:
        setup_outs(), pred()
        
        used by:
        manipulate()
        
        returns:
        allobjs (list of list of str): The objects identified at each manipulation parameter value
        allconfs (list of list of float): The prediction confidence at each manipulation parameter value
        params (list of float): The manipulation parameter values
        """
        allobjs = []
        allconfs = []
        params = []
        for param in range(0, 360, 10):
            rotated_img = ndimage.rotate(self.img, param, axes=(0, 1))
            outs = self.setup_outs(rotated_img)
            objs, confs, _ = self.pred(outs)
            allobjs.append(objs)
            allconfs.append(confs)
            params.append(param)
            
        return allobjs, allconfs, params
    
    
    def noise(self):
        """
        Add gaussian noise by adjusting the noise standard deviation
        
        uses:
        setup_outs(), pred()
        
        used by:
        manipulate()
        
        returns:
        allobjs (list of list of str): The objects identified at each manipulation parameter value
        allconfs (list of list of float): The prediction confidence at each manipulation parameter value
        params (list of float): The manipulation parameter values
        """
        allobjs = []
        allconfs = []
        params = []
        for param in np.arange(0, 10.1, 0.5):
            gauss = np.random.normal(0, param, self.img.shape)
            gauss = gauss.reshape(self.img.shape[0],
                                  self.img.shape[1],
                                  self.img.shape[2]).astype('uint8')
            img_gauss = cv.add(self.img, gauss)
            outs = self.setup_outs(img_gauss)
            objs, confs, _ = self.pred(outs)
            allobjs.append(objs)
            allconfs.append(confs)
            params.append(param)
            
        return allobjs, allconfs, params
    
    
    def manipulate(self, imgpth):
        """
        Stress test YoloV3 by iteratively manipulating a candidate image.
        
        uses:
        shrink(), rotate(), noise(), plotdeg()
        
        parameters:
        imgpth (str): the path to the candidate image
        """
        self.img = cv.imread(imgpth)
        meass = ['scale factor', 'angle', 'std']
        activities = ['Size Reduction', 'Rotation', 'Noise Level']
        for meas, activity in zip(meass, activities):
            print('\nprocessing {}...'.format(activity))
            height = self.img.shape[0]
            width = self.img.shape[1]
            if activity == 'Size Reduction':
                allobjs, allconfs, params = self.shrink()
            elif activity == 'Rotation':
                allobjs, allconfs, params = self.rotate()
            elif activity == 'Noise Level':
                allobjs, allconfs, params = self.noise()
            self.plotdeg(allobjs, allconfs, params, meas, activity)
            
            