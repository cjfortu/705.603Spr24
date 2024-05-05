# Email Decision Agent
An agent that determines which emailing action to take for a given customer and date.  The determination is based on highest prediction conversion (valid responses / emails sent).

To run the model and services, you can build a docker image from *Dockerfile* and run the container.  For a preview simply run all cells in *deploybook.ipynb*.

Note, *dffl.bin* is excluded due to its file size.  The *model_analysis.ipynb* recreates these files.  They are also included in the Docker image.

**docker instructions**
1) with */RL-main/Assignment* local and at the current directory, build the image locally:  
*docker build -t "cjfortu/705.603spr24:assignment14_1"  .*  
2) now run a container based on the image:  
*docker run --network=host --gpus all -it -p 8786:8786 -v /output:/output cjfortu/705.603spr24:assignment14_1*  
3) see the URL information below to access the microservices at port 8786  

## Microservice Info
**Provides two microservices**
1) returns testing performance on the email & customer data - http://localhost:8786/performance
2) returns predictions on a new customer and date:
* first send the customer data and date via POST request to http://localhost:8786/post
* then run the prediction and see the result at http://localhost:8786/predict

**Details on predictions**  
**Format for CSV:** Gender,Age,CustomerType,Email,Tenure,Date  
**example:** M,35,C,ajf3490j@msn.com,2,2024-05-06  
see **/sample_customers/sendcust.csv**  
We recommend *postman* for the POST request.

## Feature Selection & Processing
See */Analysis/exploratory_data_analysis.ipynb* to see how the data was assessed for machine learning suitability, and how the problem was posed as an RL problem. 

See *Data_Pipeline.py* for the code which manipulates and parses the data for machine learning.  

## Model Selection  
See */Analysis/model_analysis.ipynb* to see how the the RL agent was tuned and trained using Q-learning.  **The Q table is presented here.**

***See SystemsPlan.md* for a detailed discussion of Systems Planning and Requirements.** 
