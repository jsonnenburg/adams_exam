###############################################################################
# METHODS RELATED TO PREPROCESSING OF NON-TEXT FEATURES AND REVIEWS ###########
###############################################################################

import datetime
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.decomposition import PCA

from collections import Counter

import fasttext

def cols_to_int(df):
    """
    Convert features that only take on integer values to the appropriate
    data type.
    """
    int_cols = ['host_id', 
            'host_total_listings_count',
            'review_scores_rating', 
            'review_scores_accuracy', 
            'review_scores_cleanliness',
            'review_scores_checkin',
            'review_scores_communication',
            'review_scores_location',
            'review_scores_value' 
           ]
    
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce')
    
    return df


def clean_amenities(df):
    """
    Remove any curly braces and quotation marks.
    """
    df['amenities'] = df['amenities'].apply(lambda x: x.replace('{', '').replace('}', '').replace('"', ''))

    return df


def get_top_amenities(df, sorted_by='abs_corr_with_target'):
    """
    Get most interesting amenities (in terms of specified measure).
    Returned dataframe is limited to include only the 75 most frequently occuring amenities,
    and only those amenities which occur at least 5000 times are considered.
    """
    df_temp = pd.DataFrame(df.price)

    top75_amenities_full_string = list(df.amenities.apply(lambda x: x.split(',')).explode().value_counts()[0:75].index)

    for i in range(len(top75_amenities_full_string)):
        df_temp[top75_amenities_full_string[i]] = [int(top75_amenities_full_string[i] in a) for a in df.amenities]

    top75 = pd.DataFrame({'amenity': top75_amenities_full_string, 'mean_price': 0, 'std_price': 0, 'dev_from_mean': 0, 'abs_corr_with_target': 0, 'included_in': 0})

    for a in top75_amenities_full_string:
        top75.loc[(top75['amenity'] == a),'mean_price'] = df_temp.groupby(a)['price'].mean()[1].round(2)
        top75.loc[(top75['amenity'] == a),'std_price'] = df_temp.groupby(a)['price'].std()[1].round(2)
        top75.loc[(top75['amenity'] == a),'dev_from_mean'] = abs(df_temp.groupby(a)['price'].mean()[1].round(2) - df_temp.groupby(a)['price'].mean()[0].round(2))
        top75.loc[(top75['amenity'] == a),'abs_corr_with_target'] = abs(df_temp[a].corr(df_temp['price']).round(2))
        top75.loc[(top75['amenity'] == a),'included_in'] = df_temp.groupby(a).size()[1]

    top75 = top75[top75['included_in'] >= 5000].sort_values(sorted_by, ascending=False)
    
    return top75


def create_amenity_dummies(df, selection):
    """
    Create a dummy column for each of the selected amenities.
    
    @param selection - The select 'top-n' dataframe (including only the selected
                                                     relevant amenities)
    """
    top_amenities = ['tv',
       'dishwasher', 
       'dryer', 
       'lock_on_bedroom_door', 
       'private_entrance',
       'family_kid_friendly',
       'coffee_maker', 
       'iron', 
       'oven', 
       'stove',
       'bathtub',
       'dishes_and_silverware',
       'hair_dryer',         
       'hosting_amenity_50',
       'washer',
       'kitchen',
       'microwave',
       'cable_tv',
       'long_term_stays_allowed']
    
    df.amenities = df.amenities.apply(lambda x: x.split(','))
    
    top_amenities_full_string = list(selection['amenity'].values)

    for i in range(len(top_amenities)):
        df[top_amenities[i]] = [int(top_amenities_full_string[i] in a) for a in df.amenities]
        
    return df


def compute_days_as_host(df):
    """
    Compute new 'days_as_host' feature from 'host_since'.
    """
    # treat missing values
    df['flag_host_since_missing'] = df['host_since'].isnull().astype(int)
    df['host_since'] = df['host_since'].fillna('2000-01-01')
    
    # compute new feature
    df['days_as_host'] = df['host_since'].apply(lambda x: (datetime.date(2020,1,8) - datetime.datetime.strptime(x, "%Y-%m-%d").date()).days)
    
    # re-insert missing values
    df.loc[(df['flag_host_since_missing'] == 1), 'days_as_host'] = np.nan
    
    # drop 'host_since' column
    df = df.drop('host_since', 1)

    return df


def clean_total_listings_count(df):
    """
    For hosts with count of total listings equal to zero,
    replace zero with number of listings in dataset for the host.
    """
    temp = pd.DataFrame({'host_total_listings_count': df[df['host_total_listings_count'] == 0].groupby('host_id').size()})
    temp = temp.rename_axis('host_id').reset_index()
    
    # update values in original df
    df.set_index('host_id', inplace=True)
    df.update(temp.set_index('host_id'))
    df = df.reset_index()
    
    return df


class LanguageIdentification:
    """
    Taken from https://medium.com/@c.chaitanya/language-identification-in-python-using-fasttext-60359dc30ed0
    """
    def __init__(self):
        pretrained_lang_model = "data/lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2) # returns top 2 matching languages
        return predictions


def process_sentiment_score(df, reviews_score):
    """
    With the previously calculated average sentiment score of each listing's
    reviews, add this score to the feature dataframe, transform it to a categorical
    variable and handle missing values.
    """
    # add column for feature to df
    df['sentiment'] = np.nan
    
    # update with actual values
    temp = df.set_index('listing_id')
    temp.update(reviews_score)
    df['sentiment'] = temp['sentiment'].values
    
    if len(df) > 50000:
        print('Share of listings with sentiment score in training set:',1-(df.sentiment.isna().sum()/len(df)).round(2))
    else:
        print('Share of listings with sentiment score in test set:',1-(df.sentiment.isna().sum()/len(df)).round(2))

    # flag for missing sentiment score
    df['flag_missing_sentiment'] = df.sentiment.isnull().astype(int)
    
    # recode to categorical variable (also to handle high share of missing values)
    bins = [-2, -0.001, 0.25, 0.5, 0.75, 2]
    labels = ['negative', 'neutral', 'positive', 'very_positive', 'excellent']

    df['sentiment'] = pd.cut(df['sentiment'], bins=bins, labels=labels, include_lowest=True)
    df['sentiment'] = df['sentiment'].cat.add_categories('missing')
    df.loc[(df['flag_missing_sentiment'] == 1), 'sentiment'] = df.loc[(df['flag_missing_sentiment'] == 1), 'sentiment'].fillna('missing')
    df = df.drop('flag_missing_sentiment', 1)
    
    return df
    

def recode_neighbourhood(df):
    """
    In order to reduce the cardinality of this feature, summarize supposedly 
    similar neighbourhoods into a single category (by looking at their distance
    from the city center and the average listing price).
    """
    nbhood_dict = {
        **dict.fromkeys(['City of London', 'Westminster'], 'City of London AND Westminster'),
        **dict.fromkeys(['Kingston upon Thames', 'Hounslow'], 'Kingston upon Thames AND Hounslow'),
        **dict.fromkeys(['Hillingdon', 'Havering'], 'Hillingdon and Havering'),
        **dict.fromkeys(['Harrow', 'Barking and Dagenham', 'Bexley', 'Sutton'], 'Harrow AND Barking and Dagenham AND Bexley AND Sutton'),
        **dict.fromkeys(['Croydon', 'Redbridge'], 'Croydon AND Redbridge'),
        **dict.fromkeys(['Enfield', 'Bromley'], 'Enfield AND Bromley'),
        **dict.fromkeys(['Merton', 'Greenwich', 'Newham', 'Barnet', 'Ealing'], 'Merton AND Greenwich AND Newham AND Barnet AND Ealing'),
        **dict.fromkeys(['Waltham Forest', 'Lewisham'], 'Waltham Forest AND Lewisham')
        }
    
    nbhood_dict.update({'Tower Hamlets': 'Tower Hamlets',
                   'Hackney': 'Hackney',
                   'Camden': 'Cambden',
                   'Kensington and Chelsea': 'Kensington and Chelsea',
                   'Islington': 'Islington',
                   'Lambeth': 'Lambeth',
                   'Southwark': 'Southwark',
                   'Wandsworth': 'Wandsworth',
                   'Hammersmith and Fulham': 'Hammersmith and Fulham',
                   'Brent': 'Brent',
                   'Lewisham': 'Lewisham',
                   'Haringey': 'Haringey'})
    
    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].map(nbhood_dict)
    
    return df

def recode_property_type(df):
    """
    Recode the property types to reduce the cardinality.
    """
    property_type_dict = {
        **dict.fromkeys(['Apartment'], 'Apartment'),
        **dict.fromkeys(['House'], 'House'),
        **dict.fromkeys(['Townhouse', 'Loft'], 'Townhouse AND Loft'),
        **dict.fromkeys(['Condominium'], 'Condominium'),
        **dict.fromkeys(['Serviced apartment'], 'Serviced apartment'),
        **dict.fromkeys(['Bed and breakfast',
                         'Guest suite',
                         'Guesthouse',
                         'Other',
                         'Boutique hotel',
                         'Hotel',
                         'Hostel',
                         'Bungalow',
                         'Cottage',
                         'Boat',
                         'Tiny house',
                         'Aparthotel',
                         'Villa',
                         'Houseboat',
                         'Earth house',
                         'Cabin',
                         'Camber/RV',
                         'Hut',
                         'Yurt',
                         'Chalet',
                         'Barn',
                         'Dome house',
                         'Farm stay',
                         'Tent',
                         'Treehouse',
                         'Island',
                         'Bus',
                         'Campsite',
                         'Plane',
                         'Windmill',
                         'Parking Space',
                         'Lighthouse'], 'Other')
    }
    
    df['property_type'] = df['property_type'].map(property_type_dict)
    
    return df

def transform_zipcode(train, test):
    """
    Perform PCA on the zipcode feature and replace the original column with three
    numeric factors.
    
    Requires extracting the outward code from the zipcodes first to reduce the cardinality.
    """
    # preprocess: get outword code to drastically reduce cardinality
    train['flag_missing_zipcode'] = train['zipcode'].isnull().astype(int)
    train['zipcode'] = train['zipcode'].fillna(value='missing')
    
    test['flag_missing_zipcode'] = test['zipcode'].isnull().astype(int)
    test['zipcode'] = test['zipcode'].fillna(value='missing')
    
    test.iloc[3207, test.columns.get_loc('zipcode')] = 'missing'
    test.iloc[3207, test.columns.get_loc('flag_missing_zipcode')] = 1
        
    train['zipcode'] = train.zipcode.astype(str).apply(lambda x: x.lower().split()[0][0:4])
    test['zipcode'] = test.zipcode.astype(str).apply(lambda x: x.lower().split()[0][0:4])

    # perform PCA 
    train.zipcode = train['zipcode'].astype('category')
    test.zipcode = test['zipcode'].astype('category')

    zipcode = train[['zipcode']]
    encoder_zip = ce.BackwardDifferenceEncoder(cols=['zipcode'])
    df_zip = encoder_zip.fit_transform(zipcode)
    
    pca_zip = PCA(n_components=3).fit(df_zip)
    pca_df_zip = pca_zip.transform(df_zip)
    e_dataframe = pd.DataFrame(pca_df_zip, columns=['zip_1', 'zip_2', 'zip_3'], index=train.index)
    train[['zip_1','zip_2','zip_3']] = e_dataframe
    
    df_test_zip = encoder_zip.fit_transform(test["zipcode"])
    
    missing_zips = [zip for zip in df_zip.columns[~df_zip.columns.isin(df_test_zip.columns)].values]
    df_missing_zips = pd.DataFrame(np.zeros((len(df_test_zip), len(missing_zips))), columns=missing_zips)
    df_test_zip = pd.concat([df_test_zip, df_missing_zips], axis=1).reindex(df_test_zip.index)
    
    pca_df_test_zip = pca_zip.transform(df_test_zip)
    e_dataframe_test = pd.DataFrame(pca_df_test_zip, columns=['zip_1', 'zip_2', 'zip_3'], index=test.index)
    test[['zip_1','zip_2','zip_3']] = e_dataframe_test
    
    # drop original column
    train = train.drop('zipcode', 1)
    test = test.drop('zipcode', 1)
    
    return train, test


def analyze_missing_reviews(df):
    """
    Define flags for subgroups of missing reviews / reviews per month / review scores. 
    """
    # flag: missing all review_scores:
    df['flag_no_review_scores'] = (df['review_scores_rating'].isnull() & df['review_scores_accuracy'].isnull() & df['review_scores_cleanliness'].isnull() & df['review_scores_checkin'].isnull() & df['review_scores_communication'].isnull() & df['review_scores_location'].isnull() & df['review_scores_value'].isnull()).astype(int)

    # flag: missing reviews_per_month:
    df['flag_no_reviews_per_month'] = df['reviews_per_month'].isnull().astype(int)

    # flag: missing both all review_scores and reviews_per_month:
    df['flag_no_reviews'] = (df['flag_no_review_scores'] & df['flag_no_reviews_per_month']).astype(int)
    
    print("Data contains:")
    print("-",len(df[df['flag_no_reviews'] == 1]),"listing(s) without any reviews")
    print("-",len(df[(df['flag_no_review_scores'] == 1) & (df['flag_no_reviews_per_month'] == 0)]),"listing(s) with no review scores but documented reviews")
    print("-",len(df[(df['flag_no_review_scores'] == 0) & (df['flag_no_reviews_per_month'] == 1)]),"listing(s) with review scores but no documented reviews")
    
    return df

def handle_missing_reviews(train, test):
    """
    Using the previously defined flags, handle the missing values for the 'review'
    features in the cases when the reviews are completely missing.
    """
    train.loc[(train['flag_no_reviews'] == 1),'reviews_per_month'] = train.loc[(train['flag_no_reviews'] == 1),'reviews_per_month'].fillna(-1)
    test.loc[(test['flag_no_reviews'] == 1),'reviews_per_month'] = test.loc[(test['flag_no_reviews'] == 1),'reviews_per_month'].fillna(-1)

    review_cols = ['review_scores_rating',
                   'review_scores_accuracy',
                   'review_scores_cleanliness',
                   'review_scores_checkin',
                   'review_scores_communication',
                   'review_scores_location',
                   'review_scores_value']

    for df in [train, test]:
        for col in review_cols:
            df[col] = df[col].cat.add_categories('missing')
            df.loc[(df['flag_no_reviews'] == 1), col] = df.loc[(df['flag_no_reviews'] == 1), col].fillna('missing')
            df.loc[(df['flag_no_review_scores'] == 1), col] = df.loc[(df['flag_no_review_scores'] == 1), col].fillna('missing')
    
    return train, test
    
###############################################################################
# METHODS RELATED TO PREPROCESSING OF TEXT FEATURES ###########################
###############################################################################

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re

def remove_nonascii(text):
    """
    https://stackoverflow.com/questions/150033/regular-expression-to-match-non-ascii-characters
    
    Function to remove non-ascii characters.
    """
    return re.sub("[^\x00-\x7F]+", " ", text)

def remove_whitespace(text):
    """
    Function to remove whitespace (tabs, newlines). 
    """
    return ' '.join(text.split())

def remove_punctuation_and_casing(text):
    """
    Function to remove the punctuation, upper casing and words that include
    non-alphanumeric characters.
    """
    chars = '!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    text = text.translate(str.maketrans(chars, ' ' * len(chars)))
    return ' '.join([word.lower() for word in text.split() if word.isalpha()])

def remove_stopwords(text):
    """ 
    Function to remove stopwords. 
    """
    english_stopwords = stopwords.words('english')
    
    return ' '.join([word for word in str(text).split() if word not in english_stopwords])

def get_rare_words(text_col, threshold=2):
    """
    Get all words that occur less than n times in a text column.
    
    @param text_col - The text column.
    @param threshold - The minimum number of occurences of a word.
    
    @return rare_words - List of rare words.
    """
    split_it = text_col.apply(lambda x: Counter(x.split()).most_common())

    wordcounts = {}
    for ele in split_it:
        for word, count in ele:
            if word in wordcounts.keys():
                wordcounts[word] += count
            else:
                wordcounts[word] = count

    counts_df = pd.DataFrame({"word": list(wordcounts.keys()), "count": list(wordcounts.values())})

    rare_words = list(counts_df[counts_df['count'] < threshold]['word'])
    
    return rare_words

def remove_rare_words(text, rare_words):
    """ 
    Function to remove rare words. 
    """    
    return ' '.join([word for word in str(text).split() if word not in rare_words])

def remove_single_letters(text):
    for word in text.split():
        if len(word) == 1:
            text = re.sub(r"\b"+word+r"\b", "", text)
    return text

def get_wordnet_pos(word):
    """
    Map POS tag to first character for lemmatization.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """
    Lemmatize the words in a text.
    """
    lemmatizer = WordNetLemmatizer()
    
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(text)])

def clean_text(documents):
    """
    Function for standard NLP pre-processing including removal of whitespaces,
    non-alphanumeric characters, and stopwords.
    """
    cleaned_text = []
    
    print('Processing input array with {} elements...'.format(documents.shape[0]))
    counter = 0
    
    for text in documents:
        text = remove_nonascii(text)
        text = remove_punctuation_and_casing(text)
        text = remove_stopwords(text)
        text = remove_single_letters(text)
        text = remove_whitespace(text)
        text = lemmatize_text(text)
        
        cleaned_text.append(text)

        if (counter > 0 and counter % 5000 == 0):
            print(f'Processed {counter} rows.')
            
        counter += 1
        
    return cleaned_text

def flag_missing_text_features(df):
    df.loc[(df['name'].isna()),'flag_missing_name'] = 1
    df.loc[(df['summary'].isna()),'flag_missing_summary'] = 1
    df.loc[(df['description'].isna()),'flag_missing_description'] = 1
    
    return df


from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
import keras

class text_embedding:
    
    def __init__(self, column, output_sequence_length, embeddings_index):
        
        self.df_column = column
        self.output_sequence_length = output_sequence_length
        self.embeddings_index = embeddings_index
        self.embedding_dim = 100
        
        self.voc = None
        self.word_index = None
        self.num_tokens = None
        
        self.embedding_matrix = None
        
    def embedding_prep(self):
        """
        Function to perform vectorization, obtain vocabulary and word index.
        """
        vectorizer = TextVectorization(output_sequence_length=self.output_sequence_length)
        vectorizer.adapt(self.df_column)
        
        voc = vectorizer.get_vocabulary()
        
        word_index = dict(zip(voc, range(len(voc))))
        
        self.voc = voc
        self.word_index = word_index
        
        return vectorizer

    def get_embedding_matrix(self):
        """
        Prepare embedding matrix from pretrained embeddings and variable-specific vocabulary.
        """
        self.num_tokens = len(self.voc) + 2
        
        embedding_matrix = np.zeros((self.num_tokens, self.embedding_dim))
        
        hits = 0
        misses = 0
        
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                
        print(f"Converted {hits} words ({misses} misses).")
        
        self.embedding_matrix = embedding_matrix

    def get_embedding_layer(self, trainable=False):
        """
        Function for obtaining the embedding layer.
        """
        embedding_layer = Embedding(
            self.num_tokens,
            self.embedding_dim,
            embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
            trainable=trainable,
        )
        
        print('Successfully created embedding layer.')
        
        return embedding_layer
