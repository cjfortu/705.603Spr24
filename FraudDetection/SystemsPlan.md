# Systems Plan
## MOTIVATION
### Why are we solving this problem?
**Problem Statement**  
The fraud detection model performance has deteriorated unnacceptably.  Precision is around 40% and recall is around 70%, resulting in an F1 around 0.509.

**Value Proposition**  
The CEO requests a new prototype model which improves on the current model's performance.  Such performance will also retain and possibly increase business by keeping our customers of our banks more satisfied.  

### Why is our solution a viable one?
**Does an ML solution "fit" this problem?**  
Yes, the problem involves lots of binary classified data, in which the relationships between the input and the output are not clearly known.  Also, the data can all be made either ordinal or categorical.  

**Are we able to tolerate mistakes and some degree of risk?**  
Since the credit card holder is the final claimant to whether or not a transaction was fraudulent or not, and all erroneous fraud/nofraud labels can be corrected with a financial cost, we can tolerate as much risk as we are willing to pay.  We are more sensitive to false negatives, since these are a direct cost to the company.  We are less sensitive to false positives, since a verification system can partially alleviate customer strain. 

**Is the solution feasible?**  
Yes, the data set can fit in a Pandas DataFrame, the data can be made either ordinal or categorical, and the labels can be considered truth.  

## REQUIREMENTS
### Scope
**What are our goals?**  
Our goal is a model with an F1 score over 0.95.  

**What are the success criteria?**  
If our F1 score consistently exceeds the previous F1 score of 0.509, we will consider the model successful.  

### Requirements
**What are our system assumptions?**  
The system will feed feature data for each new transaction into the model.  Fraud inferences will require additional verification by the credit card holder before the transaction is approved.  Each transaction that does not agree with the model inference initiates re-training of the model.  

**What are our system requirements?**  
The feature data must be readily available and require no human interaction.  The system must have a way of verifying the validity of a purchase with the credit card holder (mobile phone app, text, phone call, etc).

### Risks and Uncertainties
**What are the possible harms?**  
Every false negative will result in a financial/material loss, whether at the merchant, the customer, or the bank.  False positives will result in an inconvenience to the customer.  A verication system can partially alleviate these inconveniences, but over time enough false positives can turn a customer away from our banks.

**What are the causes of mistakes?**  
The data is rich, but not fully ready for machine learning.  Faulty data engineering can result in poor performance, even if an appropriate model and hyperparameters are chosen.  Thorough data exploration is necessary.  

## IMPLEMENTATION
### Development  
**Methodology**  
The **data engineering pipeline** extracts, transforms, and loads the data.  Extraction and loading are straightforward: the original data is extracted to a DataFrame and later the procesed DataFrame is loaded back to a CSV.  The transformation component includes:
* Sine/Cosine facetizing *dob* and *trans_date_trans_time* to preserve the cyclical nature of time within a day and/or day within a year.  Note that year is not considered cyclical.
* OHE *category* and *sex*.
* For *city, state, job,* and *zip* grouping into values with either a perfect conditional probability with fraud, or a conditional probability at least an order of magnitude less.  Then these values are one-hot encoded.
* Normalizing via standard scaling *amt, year,* and *dob_year*.
* Dropping *Unnamed: 0, cc_num, merchant, first, last, street, trans_num, unix_time, merch_lat, lat, merch_long, long,* and *city_pop*.  

The **data partitioning pipeline** separates the data into training, validation, and testing subsets (splits).  All datasets for the non-nfoldCV case can be extracted by the *get_training_dataset(), get_validation_dataset(),* and *get_testing_dataset()* methods.  For the nfoldCV case, the combinations of training and validation datasets can be extracted by the *get_nfoldCV_datasets* method.  

The **model metrics** capture a simple set of scores which measure binary classification performance for imbalanced datasets.  These are F1 score and ROC-AUC score.  F1 score is further broken down into precision and recall.  Recall is further borken down into sensitivity (recall of fraud points), and specificity (recall of non-fraud points).  

The **deployment strategy** is to provide a service which initiates and trains the model upon starting.  The service will return model testing performance metrics through a GET request.  Users can send single inputs in the form of a *json* file through a POST request; the prediction results will be available through another GET request.  The service source code can be cloned from a GitHub repository or a container may be built from a pullable Docker Image.  

**High-level System Design**  
The system features internal storage, service code, data handling code, and model code.  The service code will control the HTTP requests.  The internal storage will contain all raw, processed, input, and performance data.  The data handling code will perform the necessary manipulations to prepare the data for machine learning.  Lastly, the model code will train, validate, test, infer, and produce learning metrics.  

**Development Workflow**  
Data exploration is conducted first.  This drives the data handling code.  Once the data has been manipulated, validating different machine learning models takes place.  Once a model has been selected, moving all the code to different files is necessary, in order to organize by computing purpose.  The latter steps are focused on the web service.  

### Policy
**Human-Machine Interfaces**  
The web service provides a simple method for either querying model performance or getting a model prediction.  

**Regulations**  
The company would have to ensure customer data security.  A separate developer or team may have exclusive privilege to see customer PII prior to data cleansing.  

### Operations  
**Continuous Deployment**  
Each time a customer reports a transaction that does not agree with the model, the transaction should be added to the database and the model should be re-trained.  

**Post-deployment Monitoring**  
At each retrain, the system testing performance should be easily observable.  Metrics for all transactions since last retrain should also be easily observable.  

**Maintenance**  
A separate process which probes the system should be employed to ensure API functionality.  As software packages change, a parallel system could undergo upgrades until stability, then the deployed system can swap in/out.  

**Quality Assurance**  
Customer satisfaction and fraud losses should be the measures of performance for the company, whereas model performance constitutes measures of effectiveness for the company.  Both types of measures should me monitored.  