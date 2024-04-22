# Fraud Prediction
An LSTM-based time series model to predict the number of total transactions and fraudulent transactions per week up to a given date.

To run the model and services, you can build a docker image from *Dockerfile* and run the container.  For a preview simply run all cells in *deploybook.ipynb*.

**docker instructions**
1) with */TimeSeries-main/Assignment* local and at the current directory, build the image locally:  
*docker build -t "cjfortu/705.603spr24:assignment10_1"  .*  
2) now run a container based on the image:  
*docker run --network=host --gpus all -it -p 8786:8786 -v /output:/output cjfortu/705.603spr24:assignment10_1*  
3) see the URL information below to access the microservices at port 8786  

## Microservice Info
**Provides one microservice**
1) returns the predictions given a target date - http://localhost:8786/predict?date=YYYY-MM-DD
The date must be within 1 month of 2022-12-31

## Feature Selection & Processing
See */analysis/model_analysis.ipynb* to see how additional features/subfeatures were considered, and how aggregation was determined (also includes discussion of feature removal). 

See *data_pipeline.py* for the code which manipulates and parses the data for machine learning.  

## Model Selection  
See */analysis/model_analysis.ipynb* to see how the candidate machine learning models were assessed.

## Model Performance
The selected model had the following RMSE during training:
* RMSE_tot_trans = 2254.553
* RMSE_fraud_trans = 17.692
