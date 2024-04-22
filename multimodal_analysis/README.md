# Star Rating Prediction
An LSTM-based sentiment analysis model to predict a star rating from a written film review.

To run the model and services, you can build a docker image from *Dockerfile* and run the container.  For a preview simply run all cells in *deploybook.ipynb*.

Note, *amazon_movie_reviews.csv*, *dfproc.bin*, and *text_model.h5* are excluded due to their file size.  The *model_analysis.ipynb* recreates these files.  They are also included in the Docker image.

**docker instructions**
1) with */Multimodal-Analysis* local and at the current directory, build the image locally:  
*docker build -t "cjfortu/705.603spr24:assignment12_1"  .*  
2) now run a container based on the image:  
*docker run --network=host --gpus all -it -p 8786:8786 -v /output:/output cjfortu/705.603spr24:assignment12_1*  
3) see the URL information below to access the microservices at port 8786  

## Microservice Info
**Provides two microservices**
1) returns testing performance on the provided reviews - http://localhost:8786/performance
2) returns predictions on a new review:
* first send the review as a text file via POST request to http://localhost:8786/post
* then run the prediction and see the result at http://localhost:8786/predict

**Details on new reviews**  
We recommend *postman* for the POST request.

## Feature Selection & Processing
See */analysis/exploratory_data_analysis.ipynb* to see how the data was assessed for machine learning suitability.  

See *data_pipeline.py* for the code which manipulates and parses the data for machine learning.  

## Model Selection  
See */analysis/model_analysis.ipynb* to see how the the LSTM-based RNN was tuned and trained.

***See SystemsPlan.md* for a detailed discussion of Systems Planning and Requirements.** 
