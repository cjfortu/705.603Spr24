# License Plate Detection
A two-step model to extract license plate characters from traffic video or vehicle photos.

To run the model and services, you can build a docker image from *Dockerfile* and run the container.  For a preview simply run all cells in *deploybook.ipynb*.

Note that */video/LicensePlateReaderSample_4K-1.mov* and */yolov3/lpr-yolov3.weights* must be manually added due to their excess size.  They however available in the associated docker image.

**docker instructions**
1) with */ALRP* local and at the current directory, build the image locally:  
*docker build -t "cjfortu/705.603spr24:assignment8_1"  .*  
2) now run a container based on the image:  
*docker run --network=host --gpus all -it -p 8785:8785 -p 23000:23000 -v /output:/output cjfortu/705.603spr24:assignment8_1*  
3) see the URL information below to access the microservices at port 8786  

## Microservice Info
**Provides two microservices**
1) returns testing performance on the provided photos - http://localhost:8785/performance
2) returns predictions on the ingested video - http://localhost:8786/post

**Details on streaming video**
The video stream must be launched from a separate terminal using *ffmpeg -i LicensePlateReaderSample_4K-1.mov -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000* from the video directory

*ffmpeg-python* is required.

## Feature Selection & Processing
See */analysis/exploratory_data_analysis.ipynb* to see how the data was assessed for machine learning suitability.  

See *dataset.py* and *data_pipeline.py* for the code which manipulates and parses the data for machine learning.  

## Model Selection  
See */analysis/model_performance.ipynb* to see how the candidate machine learning models were assessed.  Note there is further data manipulation between the two model steps.

## Model Performance
The selected 2-model system had the following performance:
Yolo-tiny bbox testing results on 120 frames of video:
mean IOU: 0.5844170157371271
images with extraneous boxes: 18
images lacking boxes: 25

Tesseract-OCR testing results on the provided vehicle photos:
LCS mean, all plates: 0.5809073464714272
LCS mean, only if chars detected: 0.8542755095168046
proportion perfect LCS, all plates: 0.26222222222222225
proportion perfect LCS, only if chars detected: 0.38562091503267976


***See SystemsPlan.md* for a detailed discussion of Systems Planning and Requirements.** 
