import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset:

    #Class describing data sets used for this project

    DATAFILE = 'data/ozan_p_pApply_intern_challenge_03_20_min.csv'

    '''
    -------------------
    ANALYSIS I
    -------------------
    1) title proximity tfidf:       Measures the closeness of query and job title.
    2) description proximity tfidf: Measures the closeness of query and job description.
    3) main query tfidf:            A score related to user query closeness to job title and job description.
    4) query jl score:              Measures the popularity of query and job listing pair.
    5) query title score:           Measures the popularity of query and job title pair.
    6) city match:                  Indicates if the job listing matches to user (or, user-specified) location.
    7) job age days:                Indicates the age of job listing posted.
    8) apply:                       Indicates if the user has applied for this job listing.
    Feature Columns : 1-7
    Output(Class) Column: 8
    '''

    '''
    -------------------
    ANALYSIS II
    -------------------
    1) title proximity tfidf:       Measures the closeness of query and job title.
    2) description proximity tfidf: Measures the closeness of query and job description.
    3) main query tfidf:            A score related to user query closeness to job title and job description.
    4) query jl score:              Measures the popularity of query and job listing pair.
    5) query title score:           Measures the popularity of query and job title pair.
    6) city match:                  Indicates if the job listing matches to user (or, user-specified) location.
    7) job age days:                Indicates the age of job listing posted.
    8) u id:                        ID of user
    9) mgoc id:                     Class ID of the job title clicked.
    10) apply:                      Indicates if the user has applied for this job listing.
    Feature Columns : 1-7, 9-10
    Output(Class) Column: 8
    '''

    def __init__(self):
        pass


    """ Gets input features for data set """

    @staticmethod
    def get_input_features_name(analysis):
        return ['title_proximity_tfidf', 'description_proximity_tfidf', 'main_query_tfidf', 'query_jl_score', \
                      'query_title_score', 'city_match', 'job_age_days','apply'] if analysis == 1 \
            else ['title_proximity_tfidf', 'description_proximity_tfidf', 'main_query_tfidf', 'query_jl_score', \
                      'query_title_score', 'city_match', 'job_age_days','u_id', 'mgoc_id','apply']



    """ Gets train data and output class data from data file """

    @staticmethod
    def get_train_data_from_file(analysis):
        data = pd.read_csv(Dataset.DATAFILE)
        #print('Total Data Size', data.size)
        train_data = data.loc[data['search_date_pacific'] != '2018-01-27']
        input_features = Dataset.get_input_features_name(analysis)
        #print('Train Data Size', train_data.size)
        #print input_features
        train_data = train_data.apply(LabelEncoder().fit_transform)
        return train_data[input_features]

    """ Gets test data and output class data from data file """

    @staticmethod
    def get_test_data_from_file(analysis):
        data = pd.read_csv(Dataset.DATAFILE)
        test_data = data.loc[data['search_date_pacific'] == '2018-01-27']
        #print("Test Data Size", test_data.size)
        input_features = Dataset.get_input_features_name(analysis)
        #print input_features
        test_data = test_data.apply(LabelEncoder().fit_transform)
        return test_data[input_features]

    """ Read data from data file """

    @staticmethod
    def read_data():
        features = ['title_proximity_tfidf', 'description_proximity_tfidf', 'main_query_tfidf', 'query_jl_score', \
         'query_title_score', 'city_match', 'job_age_days','apply']
        data = pd.read_csv(Dataset.DATAFILE)
        return data[features]