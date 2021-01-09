import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv('disaster_messages.csv')
    categories_df = pd.read_csv('disaster_categories.csv')
    return messages_df.merge(categories_df, how='left', on='id').set_index('id')

def clean_data(df):
    categories_df = df.categories.str.split(';', expand=True)
    first_row = categories_df.iloc[0]
    colnames = first_row.map(lambda x: x.split('-')[0])
    print(colnames.head())
    categories_df.columns = colnames
    for column in categories_df:
        categories_df[column] = categories_df[column].map(lambda x: x.split('-')[1]).astype(int)
    df.drop('categories', axis=1, inplace=True)
    return df.merge(categories_df, how='left', left_index=True, right_index=True)

def save_data(df, database_filename, if_exists):
    table_name='messages'
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, if_exists=if_exists, index=True)


def main():
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