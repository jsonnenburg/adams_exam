import pandas as pd
import preprocessing as pre

def flag_missing_text_features(df):
    """
    For each of the listing name, summary, and description, set a flag if a 
    value is missing.
    """
    df['flag_missing_name'] = df['name'].isnull().astype(int)    
    df['flag_missing_summary'] = df['summary'].isnull().astype(int)
    df['flag_missing_description'] = df['description'].isnull().astype(int)
    
    return df

if __name__ == '__main__':
    
    X_train_text = pd.read_csv('data/raw/X_train_text_raw.csv', index_col='listing_id')
    X_test_text = pd.read_csv('data/raw/X_test_text_raw.csv', index_col='listing_id')
    
    X_train_text = flag_missing_text_features(X_train_text)
    X_test_text = flag_missing_text_features(X_test_text)
    
    X_train_text = X_train_text.fillna('missing')
    X_test_text = X_test_text.fillna('missing')
    
    # clean / preprocess the relevant text features
    print('Processing training set...')
    X_train_text.iloc[:, 0:3] = X_train_text.iloc[:, 0:3].apply(lambda x: pre.clean_text(x))

    # handle newly created missing values (due to preprocessing removing non-ASCII letters etc)
    X_train_text.loc[(X_train_text['name'].isna()),'flag_missing_name'] = 1
    X_train_text.loc[(X_train_text['summary'].isna()),'flag_missing_summary'] = 1
    X_train_text.loc[(X_train_text['description'].isna()),'flag_missing_description'] = 1
    
    X_train_text = X_train_text.fillna('missing')
    
    print('Preliminary preprocessing finished, removing rare words...')
    # additionally: remove rare words
    rare_words = X_train_text.iloc[:, 0:3].apply(lambda x: pre.get_rare_words(x, threshold=10))
    rare_words_name = rare_words[0]
    rare_words_summary = rare_words[1]
    rare_words_description = rare_words[2]

    X_train_text['name'] = X_train_text['name'].apply(lambda x: pre.remove_rare_words(x, rare_words_name))
    X_train_text['summary'] = X_train_text['summary'].apply(lambda x: pre.remove_rare_words(x, rare_words_summary))
    X_train_text['description'] = X_train_text['description'].apply(lambda x: pre.remove_rare_words(x, rare_words_description))

    print('\n')
    print('Processing test set...')
    X_test_text.iloc[:, 0:3] = X_test_text.iloc[:, 0:3].apply(lambda x: pre.clean_text(x))
    
    X_test_text.loc[(X_test_text['name'].isna()),'flag_missing_name'] = 1
    X_test_text.loc[(X_test_text['summary'].isna()),'flag_missing_summary'] = 1
    X_test_text.loc[(X_test_text['description'].isna()),'flag_missing_description'] = 1
    
    X_test_text = X_test_text.fillna('missing')
    
    print('Preliminary preprocessing finished, removing rare words...')
    X_test_text['name'] = X_test_text['name'].apply(lambda x: pre.remove_rare_words(x, rare_words_name))
    X_test_text['summary'] = X_test_text['summary'].apply(lambda x: pre.remove_rare_words(x, rare_words_summary))
    X_test_text['description'] = X_test_text['description'].apply(lambda x: pre.remove_rare_words(x, rare_words_description))

    X_train_text.to_csv('data/processed/X_train_text_processed.csv')
    X_test_text.to_csv('data/processed/X_test_text_processed.csv')

    print('Finished preprocessing and saved files.')