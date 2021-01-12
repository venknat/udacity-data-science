import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """
    Loads data from .csv files containing message contents and category labels
    :param messages_filepath: file containing messages from figure 8
    :param categories_filepath: file containing categories corresponding to these messages
    :return: pandas dataframe merging data from these two sets together
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    # Not much point keeping unlabeled messages, or categories without a matching message
    return messages_df.merge(categories_df, how='inner', on='id').set_index('id')


def classify_label(label):
    """
    Classifies a label as 0 or 1, assuming label is formatted as "label-[digit]"
    Any value other than 0 after the hyphen is interpreted as a 1 (the dataset contains some
    cases where a 2 is there)
    :param label: string formatted as "label-value"
    :return: 0 or 1
    """
    return 0 if label.split('-')[1] == '0' else 1


def clean_data(df):
    """
    Takes the "raw" data loaded from the csv files and returns a dataframe organized os that each category
    is in its own column.
    :param df: pandas dataframe consisting of data rawly loaded from db
    :return: clean db so semi-colon separated categories are changed to separate columns, remove duplicate messages
    """
    categories_df = df.categories.str.split(';', expand=True)
    first_row = categories_df.iloc[0]
    colnames = first_row.map(lambda x: x.split('-')[0])
    categories_df.columns = colnames
    for column in categories_df:
        categories_df[column] = categories_df[column].map(classify_label).astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = df.merge(categories_df, how='inner', left_index=True, right_index=True)

    # It turns out that there are rows that have identical messages, but different labels.  However, to keep things
    # simple, will just drop duplicate messages and take the first row that had that message, but first will lowercase
    df['message'] = df['message'].str.lower()
    df = df.drop_duplicates(subset=['message'])
    print(df.shape[0])
    return df


def save_data(df, database_filename, if_exists):
    """
    Saves df to the 'messages' table of the sqllite database at database_filename
    :param df: pandas dataframe to save
    :param database_filename: filename where the database is to be persisted
    :param if_exists: passed to pandas to_sql in case database table already exists
    :return: None
    """
    table_name='messages'
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, if_exists=if_exists, index=True)


def main():
    """
    Runs the ETL pipeline to convert messages and categories csv files to a well-structured
    sqlite db.
    :return:
    """
    if len(sys.argv) >= 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        #
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))

        if len(sys.argv) == 5:
            if_exists=sys.argv[4]
        else:
            if_exists='replace'

        save_data(df, database_filepath, if_exists)
        #
        # print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()