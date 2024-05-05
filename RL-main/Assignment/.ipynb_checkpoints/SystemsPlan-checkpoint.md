# Systems Plan
## MOTIVATION
### Why are we solving this problem?
**Problem Statement**  
The data are obtained from a hypothetical email campaign by serving 3 different Subject Lines across a three month period.  The data shows a low conversion rate (under %10).

**Value Proposition**  
Improving the conversion rate can improve customer satisfaction and increase customer interaction.  This is so because customers would less often receive emails they do not care about, and customers would more often be presented with offers they are interested in.

### Why is our solution a viable one?
**Does an ML solution "fit" this problem?**  
Yes.  This problem has human labels and some kind of relationship between inputs and outputs.  The inputs are quantifiable, as are the labels.  As an RL problem, the state space could be the customer and date, and the action space could be the email decision.  Reward could be derived from valid responses.

**Are we able to tolerate mistakes and some degree of risk?**  
Yes.  The worst that can happen are customers ignoring or unsusbscribing to emails, and the company relying on other means of advertisement to reach customers.

**Is the solution feasible?**  
Yes, traditional Q learning is possible given an appropriately sized state space, and deep Q learning is available if the state space is too large.

## REQUIREMENTS
### Scope
**What are our goals?**  
The goal will be an agent that determines the best action to take for a given customer and date, in order to reach a target conversion.

**What are the success criteria?**  
Success will be if the conversion rate significatly exceeds the original rate of 0.0925.

### Requirements
**What are our system assumptions?**  
User data and dates will pipe into the system.  The system can be delivered via web service, and receive GET and POST requests.  The system must demonstrate performance on a testing dataset.  The system will not be able to accomodate fully training an agent while deployed.

**What are our system requirements?**  
The system must be able to load pre-trained q-tables and processed email success/failure data.  The system should have periodic audits of results.

### Risks and Uncertainties
**What are the possible harms?**  
Poor email decisions could irritate customers and ignore otherwise interested customers.  This may result in some business loss to competitors or force reliance on other means of customer outreach.

**What are the causes of mistakes?**  
The mapping of customer interests to email subject lines may not be sufficiently contained in the data.

## IMPLEMENTATION
### Development  
**Methodology**  
The **data engineering pipeline** extracts, transforms, and loads the data.  Extraction and loading are straightforward: a single customer & date is read from an inbound file, or a batch of data is read from the database.  The transformation component includes:
* binary encoding the customer data and conversion to a customer type integer
* binary encoding the date type and conversion to a date type integer
* encoding the customer and date data into an integer state
Due to simplifications and streamlining, these tasks were folded with partitioning tasks into a single **data pipeline**.

The **data partitioning pipeline** acquires the evaluation data set.  Due to simplifications and streamlining, these tasks were folded with data engineering tasks into a single **data pipeline**.

**model metrics** are folded into the *reward* computations in the environment, and in the evaluation loop of the *agent*.  This is so due to the nature of reinforcement learning; rewards are computed with each step on the fly, and often packaged/presented with each episode.

Rewards were designed to acheive a specified conversion.  For each step, the agent may send an email or pass.  If the email subject is successful (meets a conversion requirement), then the agent can accrue positive reward.  If not successful, the agent will receive a negative reward.  The agent can avoid a negative penalty by passing.  Hence, the Q values will adjust to the highest conversion action, as long as the conversion meets a minimum requireent.  Otherwise the Q values will adjust to the pass option.  **The Q-table is presented in Analysis/model_performance.ipynb**

**deployment strategy**
Due to the size of the training data and the resource/time demands of training an Q-learning agent, the system will use a pre-trained model, a preprocessed email success/failure database, and a pre-derived usertype:customer_IDs hashing.  When the service is launched, these will be loaded immediately.  GET and POST requests then become available.  GET requests demonstrate testing performance, and POST requests enable a prediction for a given customer/date.  The service source code can be cloned from a GitHub repository or a container may be built from a pullable Docker Image.  

**High-level System Design**
The system is a deployed group of software that receives customer/date information and determines which action to take.  It deploys with a pretrained model, a preprocessed email success/failure database, and a pre-derived usertype:customer_IDs hashing.

The system features internal storage, service code, data handling code, and model code.  The service code will control the HTTP requests.  The internal storage will contain all raw, processed, input, and performance data.  The data handling code will perform the necessary manipulations to prepare the data for machine learning.  Lastly, the model code will show testing performance and process a customer/date.  

**Development Workflow**  
Data exploration is conducted first.  This drives the data handling code.  The email and customer data is analyzed.  The features are binary encoded and the response emails are mapped to the sent emails.  Once the data is ready for machine learning, RL hyperparameter tuning takes place.  After hyperparameters are selected, a final model is trained.  The trained model, success/failure data, and usertype:customer_IDs hashing are saved.  Lastly, we move all the code to different files in order to organize by computing purpose.  The latter steps are focused on the web service.

### Policy
**Human-Machine Interfaces**  
The web service provides a simple method for either querying model performance or getting a model prediction.

**Regulations**  
The company would have to ensure customer data security.  A separate developer or team may have exclusive privilege to view the original PII, such as the email addresses.

### Operations  
**Continuous Deployment**  
The system should ingest and process new customer/date points immediately.   

**Post-deployment Monitoring**  
Performance should be monitored for drift.  If this occurs, it may be an effect of customer motivates not being present in the data.  Data discovery and unsupervised/semi-supervised methods may be appropriate at this point.

**Maintenance**  
A separate process which probes the system should be employed to ensure API functionality.  As software packages change, a parallel system could undergo upgrades until stability, then the deployed system can swap in/out.

**Quality Assurance**  
Again data extraction may be appropriately done by a separate organization.  Also the indicators of the abovementioned emergence of true human sentiment should be determined.
