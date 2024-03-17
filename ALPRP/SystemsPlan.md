# Systems Plan
## MOTIVATION
### Why are we solving this problem?
**Problem Statement**  
The Department of Transforpation (DoT) is interested in an Automated License Plate Recognition (ALPR) system, to modernize toll collection.

**Value Proposition**  
Eliminating manual collection will improve traffic flow through tollway entry/exit points.  

### Why is our solution a viable one?
**Does an ML solution "fit" this problem?**  
Yes, the problem involves lots of binary classified data, in which the relationships between the input and the output are not clearly known.  Also, the data can all be made either ordinal or categorical.  

**Are we able to tolerate mistakes and some degree of risk?**  
No driver will want to receive bills intended for someone else.  Hence we are more sensitive to false positives.

**Is the solution feasible?**  
Yes, given the ability to improve the component license plate cropping and character recognition models.

## REQUIREMENTS
### Scope
**What are our goals?**  
The final goal will be a system requiring no human QA/QC or retraining.  An interim developmental goal will be a system that can handle live streaming video and make correct predictions more often than not.

**What are the success criteria?**  
A goal for a system requiring no human QA/QC or retraining, will be a SequenceMatcher mean score of 1.0 across 24hrs of traffic video.  A goal for a system that shows developmental promise will be a SequenceMatcher mean score above 0.5 across a sample of traffic video.

### Requirements
**What are our system assumptions?**  
There are live traffic video streams with known video attributes (resolution, fps).  We can use a one or two step model for predictions.

**What are our system requirements?**  
The system requires human QA/QC on both the quality of bounding box detection and OCR.  The system cannot proceed to automated billing without human QA/QC.  The system requires prior knowledge of optimal regions within a frame, a desired frame rate, and will accept streams that are broken into segments (i.e. 60 seconds).  The system must have the ability to swap component models.  The system must have the ability to further manipulate data between models.

### Risks and Uncertainties
**What are the possible harms?**  
False positives will generate enough complaints to create more workload.  False negatives will negate the purpose of tollways, which is to both control traffic flow and generate revenue.

**What are the causes of mistakes?**  
OCR performance depends on the quality of the inbound image.  The text must be properly oriented, not warped, ascertainable, and cropped very close to the characters.  The bounding box model (Yolo in this case) performs these tasks.  Further, the model must be fast enough to handle streaming data, so a powerful bboxing model may be too slow.  Aside from the model, decals and coloration on the plate can lead to model mistakes.

## IMPLEMENTATION
### Development  
**Methodology**  
The **data engineering pipeline** extracts, transforms, and loads the data.  Extraction and loading are straightforward: data is read from a UDP video stream and loaded as a numpy array .npy file.  The transformation component includes:
* Reducing the frame rate to 2fps.  This is done to accomodate model speed.
* Slicing the frames down to regions where plates are ascertainable.  This is also done to accomodate model speed.

The **data partitioning pipeline** separates the data into training, validation, and testing subsets (splits).  There two sets of input data, *Ximg* for the full vehicle images, and *Xcrp* for the license plate images.  There are two sets of label data, *Ybox* for the bounding box values, and *Yplt* for the plate characters.  All datasets for the non-nfoldCV case can be extracted by the *get_training_dataset(), get_validation_dataset(),* and *get_testing_dataset()* methods.  For the nfoldCV case, the combinations of training and validation datasets can be extracted by the *get_nfoldCV_datasets* method.  The *get_full_dataset()* method retrieves all data.  

The **model metrics** captures scores which measure both the performance output of Yolo and the performance output of Tesseract-OCR.  For Yolo, we use mean IOU, the # of occurences of extraneous boxes detected, and the # of occurences of boxes lacking.  Mean IOU directly measures the overlap quality of the predicted bbox vs the truth bbox.  The other two metrics measure the frequency of box errors (extra boxes or boxes missing).  For Tesseract-OCR, we use the SequenceMatcher (SM) metric, which is algorithmically based on LCS (least common subsequence).  The mean SM score is measured across all plates, and across plates with characters detected.  The proportion of perfect SM scores (indicated perfect matches with truth) is measured across all plates, and across plates with characters detected.  SM score measures the quality of a string match from 0 to 1.

**deployment strategy** 
The service will receive a video stream and initiate a model upon starting.  Once the video stream is received, GET requests become available.  The service will return model testing performance metrics through a GET request.  The service will also process the received video and generate predictions through a different GET request.  The results from both testing performance and video prediction will be available through outputted text files.  The service source code can be cloned from a GitHub repository or a container may be built from a pullable Docker Image.  

Between models, the license plates crops are grayscaled, deskewed, color inverted, and stripped of extraneous characters (whitespace, nonalphanumeric, and lowercase).

Alternate to a single deployed system could be parallel systems, each focused on a single lane of traffic; this could allow better Yolo and OCR performance while maintaining prediction speed.

**High-level System Design**
The system is a deployed group of software that receives traffic video and produces license plate detections.  It deploys with human inputted parameters for incoming video, and may output to either a separate billing system or a human QA/QC pipeline.

The system features internal storage, service code, data handling code, and model code.  The service code will control the HTTP requests.  The internal storage will contain all raw, processed, input, and performance data.  The data handling code will perform the necessary manipulations to prepare the data for machine learning.  Lastly, the model code will show testing performance and process a video stream.  

**Development Workflow**  
Data exploration is conducted first.  This drives the data handling code.  Both the provided vehicle photographs and a sample video stream are analyzed.  The photographs are fully annoted for both bounding box and plate character truths.  Once the data has been manipulated, validating different machine learning models takes place.  This includes both the bbox and OCR models.  We will consider the data manipulations between models to be part of the two step model.  Once models have been selected, moving all the code to different files is necessary, in order to organize by computing purpose.  The latter steps are focused on the web service.  

### Policy
**Human-Machine Interfaces**  
The web service provides a simple method for either querying model performance or getting a model prediction.  The system may output to a separate billing mechanism, or perhaps more appropriately output to a human QA/QC mechanism.  Further, the system requires human knowledge of the expected video stream.

**Regulations**  
The company would have to ensure driver data security.  A separate developer or team may have exclusive privilege to audit license plate detections.  

### Operations  
**Continuous Deployment**  
The system should ingest and process segments of video at a time.  Perhaps 60s segments.    

**Post-deployment Monitoring**  
The system should output to human QA/QC at this time.  While there may be ways to deal with repeated but differing detections of the same plate, the priority should perhaps be in improving the models, the data transformations, and the ways the models are used.

**Maintenance**  
A separate process which probes the system should be employed to ensure API functionality.  As software packages change, a parallel system could undergo upgrades until stability, then the deployed system can swap in/out.

**Quality Assurance**  
Again license plate auditing may be appropriately done by a separate organization.  Aside from assuring correct billing, the results would also be used to improve model performance.
