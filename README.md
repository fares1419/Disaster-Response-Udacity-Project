# Disaster Response Pipeline Project
### Project Motivation
In this project, I build website that asks user to enter message then classify it by ML model and NLP tools.

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- DisasterResponse.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md    

### Components
There are three components I completed for this project. 

#### 1. ETL Pipeline
process_data.py is a pipeline for cleaning data:

 - Loads CSVs data
 - Cleans the data
 - Stores it in a SQLite database
 

 
#### 2. ML Pipeline
 train_classifier.py is a machine learning pipeline :

 - Loads data from the database
 - Splits the dataset into training and test sets
 - Builds a text processing 
 - Trains and tunes a model using GridSearchCV
 - Predict on test data
 - Exports the final model as a pickle file

#### 3. Flask Web App
User enter the message and model classify it



###How to run the Project ?:
1. Run the following commands in the project's root directory to set up your database and model.

    - Run ETL pipeline by:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run ML pipeline by:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for starter code for the web app. 
