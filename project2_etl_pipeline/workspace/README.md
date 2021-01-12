# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`.
      By default this will delete the existing `messages` table in the database file, to override this
      an additional argument can be added that should match a valid value to the `if_exists`
      parameter of `pandas.DataFrame.to_sql`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`  Note that this file is hardcoded to expect the disaster data in 
   `data/disaster_message.db`, in table `messages`.  

3. Go to http://0.0.0.0:3001/
