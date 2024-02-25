# Fraud Detection
An XGBoost Classifier trained to predict whether a credit card transaction is fraudulent or not.

To run the model and services, you can build a docker image from *Dockerfile* and run the container.  For a preview simply run all cells in *deploybook.ipynb*.

Note that *transactions-1.csv* must be manually added to */data/* due to its excess size.  It is however available in the associated docker image.

**docker instructions**
1) with */FraudDetection* local and at the current directory, build the image locally:  
*docker build -t "cjfortu/705.603spr24:assignment5_1"  .*  
2) now run a container based on the image:  
*docker run --gpus all -it -p 8785:8785 -v /output:/output cjfortu/705.603spr24:assignment5_1*  
3) see the URL information below to access the microservices at port 8786  

## Microservice Info
**Provides two microservices**
1) returns testing performance stats - http://localhost:8786/performance
2) receives an input point via json file for prediction - http://localhost:8786/post
2) returns the prediction based on the received input point - http://localhost:8786/predict

**Details on input json**
The following format is required:
[{"trans_date_trans_time": "2019-01-02 01:06:37", "category": "grocery_pos", "amt": 286.01, "sex": "M", "city": "Collettsville", "state": "NC", "zip": 28611, "job": "Soil scientist", "dob": "1988-09-15"}]  

*Postman* is recommended to send the POST request.  The json should be attached to the *body*.

## Feature Selection & Processing
See */analysis/exploratory_data_analysis.ipynb* to see how the data was assessed for machine learning suitability.  

See *dataset.py* and *data_pipeline.py* for the code which manipulates and parses the data for machine learning.  

## Model Selection  
See */analysis/model_performance.ipynb* to see how the candidate machine learning models were assessed.

## Model Performance
The selected XGBoost Classifier has a testing performance of:
* F1=0.901
* sensitivity(fraud recall) = 0.852
* specificity(nofraud recall) = 0.999
* precision = 0.955
* ROCAUC = 0.926


***See SystemsPlan.md* for a detailed discussion of Systems Planning and Requirements.** 