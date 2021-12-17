"""
@author: kirstaylo
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logreg


class SiotML():
    def __init__(self):
        # instantiate the variable to hold the entire dataframe for full processing
        self.data = None
        # instantiate the list to hold the music data from the listening sessions
        self.music_data = {}
        self.timestamp_data = {}
        # load the csv data into self.music
        self.load_music_data()
        
        self.fields_X = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                             'instrumentalness', 'liveness', 'valence', 'tempo']
        self.fields_y = ['EDA_li', 'EDA_lr', 'HR_li', 'HR_lr', 'TEMP_li', 'TEMP_lr']
        
        # create a store of data for the threshold change in biological data to be considered a significant stimulus
        self.logreg_thresholds = {
            'EDA_li' : 0.001,
            'EDA_lr' : 0.001, 
            'HR_li' : 0.2,
            'HR_lr' : 0.2, 
            'TEMP_li' : 0.003, 
            'TEMP_lr' : 0.003
            }
        
        
    ''' method to load the spotify data '''
    def load_music_data(self):
        # iterate through the different listening sessions
        for file in os.listdir('./spotify_data'):
            # read the csvs containing music data
            session_df = pd.read_csv('./spotify_data/{}'.format(file))
            # append to the list of music data
            self.music_data[file.split('song_data_')[1].split('.')[0]] = session_df
            self.timestamp_data[file.split('song_data_')[1].split('.')[0]] = session_df[['timestamp_start', 'timestamp_end']]
                
    
    ''' method to load in the preprocessed empatica data into the music dataframe of the correct session'''
    def load_empatica_data_old(self, empatica_dict):
        # iterate through the different sessions in the self.music_data
        for session in range(len(self.music_data)):
            # find the associated timestamps for each song in the listening session and save as numpy array
            music_times = np.array(self.music_data[session]['timestamp_end']).astype(float)
            
            # iterate through the different dataframes in a certain listening session within the empatica data dictionary
            for data_type in empatica_dict['session_{}'.format(session+1)].keys():
                # extract the associated timestamps for each datatype in the empatica data
                empatica_times = np.array(empatica_dict['session_{}'.format(session+1)]['{}'.format(data_type)]['timestamp']).astype(float)
                # create empty list to store matched timestamps between dataframes
                empatica_matched = []
                
                # iterate through each time stamps in the music data
                for timestamp in music_times:
                    # find the index of the data point in the empatica data where the timestamp is matched with the music data timestamp
                    # this means finding the last datapoint before the song is changed (before the timestamp)
                    empatica_index = np.where(empatica_times < timestamp)[0][-1]
                    # add the value at this index to the list of matches times
                    empatica_matched.append(empatica_dict['session_{}'.format(session+1)][data_type][data_type.split('.')[0]][empatica_index])
                
                # add the list of matched values to the dataframe of the associated session for a certain datatype
                self.music_data[session][data_type.split('.')[0]] = empatica_matched
    
    
    ''' method to load in the preprocessed empatica data into the music dataframe of the correct session'''
    def load_empatica_data(self, empatica_dict):
        for session in self.music_data.keys():
            self.music_data[session] = self.music_data[session][:-1]
            
            for data_type in empatica_dict[session].keys():
                self.music_data[session][data_type.split('.')[0]+'_fb'] = empatica_dict[session][data_type]['flat baseline']
                self.music_data[session][data_type.split('.')[0]+'_lr'] = empatica_dict[session][data_type]['lr baseline']
                self.music_data[session][data_type.split('.')[0]+'_li'] = empatica_dict[session][data_type]['li baseline']
            
            
    ''' method to build the full data frame '''
    def build_full_df(self):
        # join together the csv files from the separate listening sessions
        self.data = pd.concat(self.music_data, ignore_index=True)
            
        
    ''' method for counting total number of datapoints'''
    def count_dp(self):
        dp = self.data.shape[0]
        print('TOTAL SONGS: {}'.format(dp))
        print('MUSIC LENGTH: {} mins'.format(dp * 0.25))
        
        
    ''' method for making the dataframe index a timestamp '''
    def timestamp_index(self):
        time_index = pd.DatetimeIndex(pd.to_datetime(self.data['timestamp_start'].astype(int), unit='s'))
        self.data = self.data.set_index(time_index)
        
        
    ''' method for turning discrete fields in the music data into binary columns'''
    def binarize_discrete_fields(self, fields=['key', 'time_signature']):
        # iterate through binary fields
        for f in fields:
            # find the unique vals in the key and time signature and sort them 
            unique_vals = self.data[f].unique()
            unique_vals.sort()
            
            # iterate through all the unique vals within a field and create its own binary df column
            for u in unique_vals:
                self.data['{}{}'.format(f, u)] = self.data[f] == u
                self.data['{}{}'.format(f, u)] = self.data['{}{}'.format(f, u)].astype(int)

        # drop the columns of the original fields
        self.data = self.data.drop(columns=fields)
        
        
    ''' method for dropping irrelevant fields that are not used for training the ML algorithm '''
    def drop_nontraining_info(self, fields=['timestamp_start', 'timestamp_end', 'name', 'artist', 'uri', 'genre']):
        self.data = self.data.drop(columns=fields)
        
        
    ''' method for splitting the data into a training and test set '''
    def make_training_test_sets(self, split1=0.2, split2=0.5):
        # split into training, test and validation sets
        self.train, other = train_test_split(self.data, test_size=split1, random_state = 42)
        self.val, self.test = train_test_split(other, test_size=split2, random_state = 42) 
        # 42 is the Answer to the Ultimate Question of Life, the Universe, and Everything
        
        
    ''' method for the standardisation of continuous fields '''
    def standardize_continuous(self, fields=None):
        if not fields:
            fields=self.fields_X
        
        # extract continuous data for standardisation
        c_train = self.train[fields]
        c_val = self.val[fields]
        c_test = self.test[fields]

        # instantiate a scaler
        scaler = StandardScaler()
        # fit the scaler to the training data
        scaler.fit(c_train)

        # use the scaler to transform the different sets of data
        s_train = scaler.transform(c_train)
        s_val = scaler.transform(c_val)
        s_test = scaler.transform(c_test)

        # put the standardised data back into the datasets
        self.train[fields] = s_train
        self.val[fields] = s_val
        self.test[fields] = s_test
        
        self.X_train = self.train.drop(columns=['BVP_fb', 'BVP_lr', 'BVP_li', 'EDA_fb', 'EDA_lr', 'EDA_li', 'HR_fb',
       'HR_lr', 'HR_li', 'TEMP_fb', 'TEMP_lr', 'TEMP_li'])
        self.X_val = self.val.drop(columns=['BVP_fb', 'BVP_lr', 'BVP_li', 'EDA_fb', 'EDA_lr', 'EDA_li', 'HR_fb',
       'HR_lr', 'HR_li', 'TEMP_fb', 'TEMP_lr', 'TEMP_li'])
        self.X_test = self.test.drop(columns=['BVP_fb', 'BVP_lr', 'BVP_li', 'EDA_fb', 'EDA_lr', 'EDA_li', 'HR_fb',
       'HR_lr', 'HR_li', 'TEMP_fb', 'TEMP_lr', 'TEMP_li'])
        
        
    ''' method to find linear regression of data '''
    def linear_regression(self, y_field, X_field=None, plot=False):
        if not X_field:
            X_field=self.fields_X
            
        X = self.train[X_field]
        y = self.train[[y_field]]
        reg = LinearRegression().fit(X, y)
        
        test_X = self.test[X_field]
        test_y = self.test[y_field]
        score = reg.score(test_X, test_y)
        print(y_field, ' linear regression score: ', score) #reg.coef_, reg.intercept_)
        
        if plot:
            for i, x_field in enumerate(X_field):
                self.plot_correlation(x_field, y_field, linreg=[reg.coef_[0][i], reg.intercept_])
        return score
    
    
    ''' method to do all the linear regression and compare scores '''
    def do_all_lin_reg(self, y_fields, X_field=None):
        if not X_field:
            X_field=self.fields_X
        # instantiate a placeholder for bestscore and what variable it's associated with
        best_score = -np.inf
        best_field = None
        
        # iterate through the chosen y_fields
        for y_f in y_fields:
            score = self.linear_regression(y_f)
            # check to see if this is the new best score and store it if it is
            if score > best_score:
                best_score = score
                best_field = y_f
        
        print(' ')
        print(' ---- ¯\_(ツ)_/¯ ---- ¯\_(ツ)_/¯ ---- ¯\_(ツ)_/¯ ----')
        print('BEST LINEAR REGRESSION CORRELATION: ', best_field)
        print('BEST LINEAR REGRESSION SCORE: ', best_score)
        print(' ---- ¯\_(ツ)_/¯ ---- ¯\_(ツ)_/¯ ---- ¯\_(ツ)_/¯ ----')
        print(' ')
    
        
    ''' method to build a support vector regressor '''
    def build_svr(self, y_field, X_field=None, c=1, e=0.2):
        if not X_field:
            X_field=self.fields_X
            
        X = self.train[X_field]
        y = self.train[[y_field]]
        
        reg = SVR(C=c, epsilon=e).fit(X, y)
        
        # use validation set to train the hyperparameters
        val_X = self.val[X_field]
        val_y = self.val[y_field]
        score = reg.score(val_X, val_y)
            
        return score, reg
        
    
    ''' method to find the best hyperparameters and best SVR model '''
    def find_best_svr_params(self, y_field):
        # create placeholder variables
        best_score = -np.inf
        best_ce_pair = None
        best_model = None
        
        # iterate through different combos of hyperparameters
        for c in range(5,100,5):
            for e in range(5,100,5):
                score, reg = self.build_svr(y_field, c=c/100, e=e/100)
                
                if score >= best_score:
                    best_score = score
                    best_ce_pair = [c/100, e/100]
                    best_model = reg
                    
        print('{} best svr score: {}, C: {}, epsilon: {}'.format(y_field, best_score, best_ce_pair[0], best_ce_pair[1]))
                
        return best_model, best_ce_pair
        
    
    ''' method to do all the SVR to find best scores and hyperparameters '''
    def do_all_svr(self, y_fields, X_field=None):
        if not X_field:
            X_field=self.fields_X
        # instantiate a placeholder for bestscore and what variable it's associated with
        best_score = -np.inf
        best_field = None
        best_ce_pair = None
        
        # iterate through the chosen y_fields
        for y_f in y_fields:
            # get the best model and hyperparameters for the field
            reg, ce_pair = self.find_best_svr_params(y_f)
            # calculate the score, this time using the test set
            test_X = self.test[X_field]
            test_y = self.test[y_f]
            score = reg.score(test_X, test_y)
            
            # check to see if this is the new best score and store it if it is
            if score > best_score:
                best_score = score
                best_field = y_f
                best_ce_pair = ce_pair
        
        print(' ')
        print(' ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ----')
        print('BEST SUPPORT VECTOR REGRESSION CORRELATION: ', best_field)
        print('CHOSEN C: {} ---- CHOSEN EPSILON: {} '.format(best_ce_pair[0], best_ce_pair[1]))
        print('BEST SUPPORT VECTOR REGRESSION SCORE: ', best_score)
        print(' ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ---- (ﾉ◕ヮ◕)ﾉ*:･ﾟ ----')
        print(' ')
         
        
    ''' method for building a logistic regression model '''
    def build_log_reg_model(self, y_train, y_val, C=0.1):
        # build the model and calculate the score using validation sets
        lg = logreg(penalty='l2', C=C).fit(self.X_train, y_train)
        score = lg.score(self.X_val, y_val)     
        return lg, score
    
    
    ''' method for finding best parameters for logistic regression model '''
    def find_log_reg_params(self, y_field, y_train, y_val):
        # create a binary session above a threshold for the field
        
        # placeholders for holding best scores
        best_score = -np.inf
        best_model = None
        best_C = None
        
        for C in range(1, 100, 5):
            lg, score = self.build_log_reg_model(y_train, y_val, C=C/100)
            if score > best_score:
                best_score = score
                best_model = lg
                best_C = C/100
                
        print('{} best logreg score: {}, C: {}'.format(y_field, best_score, best_C))
        
        return best_model, best_C
         
    
    ''' method for doing all the logistic regression '''
    def do_all_log_reg(self, y_fields):
        # instantiate a placeholder for bestscore and what variable it's associated with
        best_score = -np.inf
        best_field = None
        best_C = None
        
        # iterate through the chosen y_fields
        for y_f in y_fields:
            # binarize the y columns by looking above a particular threshold
            y_train = (self.train[y_f] > self.logreg_thresholds[y_f]).astype(int)
            y_val = (self.val[y_f] > self.logreg_thresholds[y_f]).astype(int)
            y_test = (self.test[y_f] > self.logreg_thresholds[y_f]).astype(int)
            
            # get the best model and hyperparameters for the field
            model, C = self.find_log_reg_params(y_f, y_train, y_val)
            # calculate the score, this time using the test set
            score = model.score(self.X_test, y_test)
            
            # check to see if this is the new best score and store it if it is
            if score > best_score:
                best_score = score
                best_field = y_f
                best_C = C
        
        print(' ')
        print(' ---- (╯°□°）╯︵ ┻━┻  ---- (╯°□°）╯︵ ┻━┻ ---- (╯°□°）╯︵ ┻━┻')
        print('BEST LOGISTIC REGRESSION CLASSIFICATION: ', best_field)
        print('CHOSEN C: {}'.format(best_C))
        print('BEST LOGISTIC REGRESSION SCORE: ', best_score)
        print(' ---- (╯°□°）╯︵ ┻━┻  ---- (╯°□°）╯︵ ┻━┻ ---- (╯°□°）╯︵ ┻━┻')
        print(' ')
        
    
    ''' method for plotting the correlation between 2 fields '''
    def plot_correlation(self, x_field, y_field, linreg=None):
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(25, 5))
            plt.scatter(self.data[x_field], self.data[y_field])
            plt.title('Correlation between {} and {}'.format(x_field, y_field))
            plt.xlabel(x_field)
            plt.ylabel(y_field)
            if linreg:
                x = self.data[x_field]
                y = (self.data[x_field]*linreg[0]) + linreg[1]
                plt.plot(x, y, color='m')
            plt.show()