# Systems Plan
## MOTIVATION
### Why are we solving this problem?
**Problem Statement**  
Amazon has identified a need to move beyond the traditional 5-star rating system. Recognizing the limitations and subjectivity inherent in star ratings, the company aims to develop a more nuanced and reliable method of capturing customer feedback and product quality. Your team is tasked to propose and prototype a solution to replace the 5-star system, focusing on capturing customer sentiment in a more detailed and actionable manner. 

**Value Proposition**  
Focusing only on human verbal reviews reduces incoherent ratings (high rating with negative review, and vice versa).  This approach could more effectively re-scale the rating system, since the majority of ratings are at the highest, yet a minority of films are considered of the highest quality.  Alternately, this approach can enable data discovery in the future on more sophisticated sentiment analysis, rather than a linear rating system.  

### Why is our solution a viable one?
**Does an ML solution "fit" this problem?**  
Yes.  This problem has human labels and some kind of relationship between inputs and outputs.  The inputs are quantifiable, as are the labels.  Multiclass classification is most appropriate.

**Are we able to tolerate mistakes and some degree of risk?**  
Yes.  The worst that can happen are customers ignoring the Amazon rating system, and relying on other sources to assess the quality of a film without watching it.

**Is the solution feasible?**  
Yes, sentiment analysis is a feasible task, and the data is well suited for it.

## REQUIREMENTS
### Scope
**What are our goals?**  
The goal will be a system that assigns a star rating to a verbal review, given the nuances and inconsistencies of human language.

**What are the success criteria?**  
Success will be if the predicted ratings for any given film better represent the collective human opinion of the film, than a corpus of human star ratings.  This includes being able to ascertain when a review is sarcastic, and when a review is inconsistent with the human rating.  Quantitatively, we will first look for an accuracy score beyond the trivial solution (%60.3 in this case).

### Requirements
**What are our system assumptions?**  
Text reviews will pipe into the system.  The system can be delivered via web service, and receive GET and POST requests.  The system must demonstrate performance on a testing dataset.  The system will not be able to process a raw review database and train an MLP in an acceptable amount of time.

**What are our system requirements?**  
The system must be able to load pre-trained models, a processed review database, and token-index hashings.  The system should have periodic audits of results.

### Risks and Uncertainties
**What are the possible harms?**  
Unreliable ratings could either irritate customers or simply cause them to ignore the Amazon rating system.  This may result in some business loss to competitors.

**What are the causes of mistakes?**  
Misspelled words will cause problems with the normalizing/stemming/tokenizing process.  Inconsistent reviews (high rating with negative review, or vice versa) can also impact learning.

## IMPLEMENTATION
### Development  
**Methodology**  
The **data engineering pipeline** extracts, transforms, and loads the data.  Extraction and loading are straightforward: a single review is read from an inbound file, or a batch of reviews are read from the database.  The transformation component includes:
* normalizing/stemming/tokenizing
* encoding to token indices
Due to simplifications and streamlining, these tasks were folded with partitioning tasks into a single **data pipeline**.

The **data partitioning pipeline** separates the data into training, validation, and testing sets (splits).  Due to simplifications and streamlining, these tasks were folded with data engineering tasks into a single **data pipeline**.

The **model metrics** captures scores which indicate the adherence of predictions to the human star ratings.  We select *accuracy* because it provides an intiuitive metric for whether or not the model is learning. We simply compare *accuracy* to the proportion of the most frequent rating.  If *accuracy* is greater, then the model has surpassed the trivial solution.  

We also derive a metric called *proximalperf.* This assigns a value of 1 for each correct label, and a value (5-abs(diff))/nclasses for each incorrect label. For example, if *y_truth=5* and *y_pred=2*, then the value will be 2/5.  This allows some lenience if scores were only a single star rating off.  This is important because there is inherent subjectivity and error in the star ratings versus the text.

**deployment strategy**
Due to the size of the training data and the resource/time demands of training an LSTM-based RNN, the system will use a pre-trained model, a preprocessed review database, and a pre-derived token-index hashing.  When the service is launched, these will be loaded immediately.  GET and POST requests then become available.  GET requests demonstrate testing performance, and POST requests enable a prediction for a text review.  The results from both testing performance and video prediction will be available through outputted text files.  The service source code can be cloned from a GitHub repository or a container may be built from a pullable Docker Image.  

**High-level System Design**
The system is a deployed group of software that receives review text and produces star predictions.  It deploys with a pretrained model, a preprocessed review database, and a pre-derived token-index hashing.

The system features internal storage, service code, data handling code, and model code.  The service code will control the HTTP requests.  The internal storage will contain all raw, processed, input, and performance data.  The data handling code will perform the necessary manipulations to prepare the data for machine learning.  Lastly, the model code will show testing performance and process a text review.  

**Development Workflow**  
Data exploration is conducted first.  This drives the data handling code.  The corpus of reviews are analyzed.  The reviews are cleansed, normalized, stemmed, tokenized, and encoded to token indices.  Once the data is ready for machine learning, RNN hyperparameter tuning takes place.  After hyperparameters are selected, a final model is trained.  The processed review data, the token-index hashing, and the trained model are saved.  Lastly, we move all the code to different files in order to organize by computing purpose.  The latter steps are focused on the web service.

### Policy
**Human-Machine Interfaces**  
The web service provides a simple method for either querying model performance or getting a model prediction.  The results can be manually reviewed by examining the output file.

**Regulations**  
The company would have to ensure customer data security.  A separate developer or team may have exclusive privilege extract the review and rating data to ensure the machine learning team cannot access customer PII.  

### Operations  
**Continuous Deployment**  
The system should ingest and process reviews immediately.   

**Post-deployment Monitoring**  
Performance should be monitored for drift.  If this occurs, it may be an effect of true human sentiment emerging and separating from the linear rating system.  Data discovery and unsupervised/semi-supervised methods may be appropriate at this point.

**Maintenance**  
A separate process which probes the system should be employed to ensure API functionality.  As software packages change, a parallel system could undergo upgrades until stability, then the deployed system can swap in/out.

**Quality Assurance**  
Again data extraction may be appropriately done by a separate organization.  Also the indicators of the abovementioned emergence of true human sentiment should be determined.
