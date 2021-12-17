"""
@author: kirstaylo
"""
import os
import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Empatica():
    def __init__(self):
        # instantiate the empty data dictionary
        self.data = {}
        # find the total number of listening sessions
        self.sessions = len(os.listdir('./empatica_data'))
        # load in the data for all sessions
        self.load_data()
        
        # instantiate a dictionary with the window sizes for anomaly detection and noise smoothing
        self.window_sizes = {
            'anomaly_detection' : {
                'EDA' : 7,
                'HR' : 10,
                'BVP' : 10,
                'TEMP' :10
                },
            'noise_smoothing' : {
                'EDA' : 20,
                'HR' : 2,
                'BVP' : 10,
                'TEMP' :20
                }
            }
        
        
    ''' method to load in the csv data from a particular listening session'''
    def load_session(self, session):
        # create an empty dictionary for the listening sessions
        session_dict = {}
        # iterate through the empatica data files from a particular session
        for file in os.listdir('./empatica_data/session_{}'.format(session)):
            # save the file as a pandas dataframe in the session dictionary
            session_dict[file] = pd.read_csv('./empatica_data/session_{}/{}'.format(session, file))
        # append the file to the database
        self.data['session_{}'.format(session)] = session_dict
        
        
    ''' method to load in all csv data from all listening sessions '''
    def load_data(self):
        # iterate through all the sessions and load into the data dict
        for i in range(self.sessions):
            self.load_session(i+1)
            
    
    ''' method to see the head of a certain dataframe '''
    def get_df_head(self, session, data_type):
        data_df = self.data['session_{}'.format(session)]['{}.csv'.format(data_type)]
        print(data_df.head())
    
    
    ''' method to visualise a certain section of data'''
    def visualise_data(self, session, data_type):
        plot_data_df = self.data['session_{}'.format(session)]['{}.csv'.format(data_type)]
        plot_data = plot_data_df[data_type]
        x = range(1, plot_data_df.shape[0]+1)
        plt.plot(x, plot_data, "b")
        plt.xlabel('Sample no.')
        plt.ylabel('{}'.format(data_type))
        
        
    ''' setter for window sizes '''
    def set_window_size(self, window_size, data_type, window_type):
        self.window_sizes[window_type][data_type] = window_size
    
    
    ''' method to plot a moving average on the data and find anomalous values'''
    def moving_average(self, data_type, data, window_size, find_anomalies=False, scale=5, plot=True):
        # find the rolling mean of the data
        rolling_mean = data.rolling(window=window_size).mean()
        
        if plot:
            # instantiate figure
            plt.figure(figsize=(25,5))
            # session 1 8700-8750
            plt.title("Anomaly Detection on {} data".format(data_type))
            # plot the original data
            plt.plot(data[window_size:],"b", label="Original values")
            # plot the rolling mean
            plt.plot(rolling_mean, "c", label="Rolling mean of window size {}".format(window_size))
            plt.xlabel('Sample no.')
            plt.ylabel('{}'.format(data_type))
            plt.legend(loc="upper left")
            plt.grid(True)
        
        # find and plot the confidence intervals for the smoothed values
        if find_anomalies:
            # find the mean absolute percentage error
            mae = np.mean(np.abs((data[window_size:] - rolling_mean[window_size:]) / data[window_size:]))
            deviation = np.std(data[window_size:] - rolling_mean[window_size:])
            # find the upper and lower bounds of the smoothed values
            lb = rolling_mean - (mae + (scale * deviation))
            ub = rolling_mean + (mae + (scale * deviation))
            
            if plot:
                # plot the upper and lower bounds
                plt.plot(ub, "m--", label="Upper Bound / Lower Bound")
                plt.plot(lb, "m--")
        
            # use the intervals to find anomalies 
            # find the lower anomalies
            la = data<lb
            # find the upper anomalies
            ua = data>ub
            anomalies = pd.DataFrame(index=data.index, columns=[data_type])
            # save the lower anomalies into the anomaly dataframe
            anomalies[la] = np.array(data[la]).reshape(-1,1)
            # save the upper anomalies into the anomaly dataframe
            anomalies[ua] = np.array(data[ua]).reshape(-1,1)
            
            if plot:
                # plot anomalies
                plt.plot(anomalies, "ro", markersize=7)
        
            # extract the indexes for the upper and lower anomalies
            la_i = np.where(la==True)[0]
            ua_i = np.where(ua==True)[0]
    
            return la_i, ua_i
        
        return rolling_mean
    
    
    ''' method to find and replace anomalies with lienar interpolated points '''
    def fix_anomalies(self, data_type, data, window_size, scale=5, plot=True):
        # create a copy of the dataset
        fixed_data = data.copy()
        # plot anomalies before anomaly fixing
        la_i, ua_i = self.moving_average(data_type, data, window_size, find_anomalies=True, scale=scale, plot=plot)
        # fix the anomalous results using linear interpolation
        for i in la_i:
            fixed_data.iloc[i] = (fixed_data.iloc[i-1] + fixed_data.iloc[i+1])/2
        for i in ua_i:
            fixed_data.iloc[i] = (fixed_data.iloc[i-1] + fixed_data.iloc[i+1])/2
        
        # plot the fixed data
        if plot:
            self.moving_average(data_type, fixed_data, window_size, find_anomalies=True, scale=scale, plot=plot)
            
        return fixed_data
    
    
    ''' method to fix all the anomalies in the dataset '''
    def fix_all_anomalies(self, plot=False):
        # craete a dataset without the anomalies
        self.data_na = {}
        
        # iterate through the different listening sessions
        for session in self.data.keys():
            # create a new dictionary within the no anomaly dictionary for each session
            self.data_na[session] = {}
            
            # iterate through the different datafiles for each session:
            for data_file in self.data[session].keys():
                data_type = data_file.split('.')[0]
                data = self.data[session][data_file][data_type]
                
                # find the window size for the data type
                window_size = self.window_sizes['anomaly_detection'][data_type]
                # get rid of the anomalies
                na_data = self.fix_anomalies(data_type, data, window_size, plot=plot)
                # add the fixed data to the new dictionary
                self.data_na[session][data_file] = na_data
        
    
    ''' method to do exponential smoothing on the data'''
    def exponential_smoothing(self, data, alpha):
        data = np.array(data)
        # find the first datapoint and store in a list
        results = [data[0]] # first value is same as series
        for n in range(1, len(data)):
            # appened the exponentially smoothed datapoint to the results list
            results.append(alpha * data[n] + (1 - alpha) * results[n-1])
        return results
    
    
    ''' method to plot exponential smoothing on the data'''
    def plot_exponential_smoothing(self, data_type, data, alphas):
        # plot the results from the smoothing
        with plt.style.context('seaborn-white'):    
            plt.figure(figsize=(25, 5))
            for alpha in alphas:
                plt.plot(self.exponential_smoothing(data, alpha), label="Alpha {}".format(alpha))
            plt.plot(data.values, "c", label = "Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing for {} data".format(data_type))
            plt.grid(True);
    
    
    ''' method to build a fourier analysis of the data '''
    def plot_fourier_transform(self, data_dict, data_type, freq=4):
        with plt.style.context('seaborn-white'):  
            # instantiate figure
            plt.figure(figsize=(25, 5))
            # iterate through the different datasets
            for key, data in data_dict.items():
                data = np.array(data)
                # compute the fast fourier transform
                yf = scipy.fftpack.fft(data)
                yf = yf[200:]
                # create the range of corresponding frequencies
                x = scipy.fftpack.fftfreq(yf.size, 1 / freq)
                # plot the fft
                plt.plot(x[:x.size//2], abs(yf)[:yf.size//2], label = key)
                
            plt.legend(loc="best")
            plt.vlines(0.26, 0, 5, colors='m', linestyles='dashed')
            plt.axis('tight')
            plt.title("Fourier Transform for {} data".format(data_type))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True);
            plt.show()
            
            # plot the different filters
            plt.figure(figsize=(25, 5))
            for key, data in data_dict.items():
                data = np.array(data)
                plt.plot(data[100:200], label=key)
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Moving average filters on {} data".format(data_type))
            plt.xlabel('Sample no.')
            plt.ylabel('{}'.format(data_type))
            plt.grid(True);
            plt.show()
    
    
    ''' method to smooth the data to get rid of sensor noise '''
    def smooth_data(self):
        # craete a dictionary for the smoothed values
        self.data_sm = {}
        
        # iterate through the different listening sessions
        for session in self.data_na.keys():
            # create a new dictionary within the smooth dictionary for each session
            self.data_sm[session] = {}
            
            # iterate through the different datafiles for each session:
            for data_type in self.data_na[session].keys():
                data = self.data_na[session][data_type]
                # find the correct window value from the dictionary
                window_size = self.window_sizes['noise_smoothing'][data_type.split('.')[0]]
                # smooth the data using the window and save to the new dict
                self.data_sm[session][data_type] = data.rolling(window=window_size).mean()
      
                
    def build_line_function(self, x1, x2, y1, y2):
        # find the gradient and offset
        m = (y2-y1) / (x2-x1)
        c = y1 - (m * x1)
        
        def line(x):
            return (m*x) + c
        # return the function for the line
        return line
    
    
    ''' method to find the flat baseline before a stimulus event '''
    def baseline(self, seconds, timestamp_dict, listening_session=15):
        # craete a dictionary for the baselines before a song starts
        self.baselines = {}
        # create a dictionary for the linear regression models before a song starts
        self.linreg = {}
        # create a dictionary for the linear interpolations for a particular song
        self.lininterp = {}
        
        # iterate through the different listening sessions
        for session in self.data_sm.keys():
            # create a new dictionary within the baseline dictionary for each session
            self.baselines[session] = {}
            self.linreg[session] = {}
            self.lininterp[session] = {}
            
            # iterate through the songs 
            for song_no in range(timestamp_dict[session].shape[0]):
                song = timestamp_dict[session].iloc[song_no]
                # find the start time of the song playing
                start = int(np.ceil(song['timestamp_start']))
                # minus the number of seconds to sample from to find baseline ( <10 )
                pre_start = start - seconds
            
                # iterate through the different datafiles for each session:
                for data_type in self.data_sm[session].keys():
                    
                    # find the corresponding datapoint indexes to for a particular datatype
                    times = self.data[session][data_type]['timestamp']
                    start_i = np.where(times>start)[0][0]
                    pre_start_i = np.where(times<pre_start)[0][-1]
                    
                    # find the vales in the timeframe using the indexes
                    values = self.data_sm[session][data_type][pre_start_i:start_i]
                    # find the basepoint by averaging the values
                    baseline = values.mean()
                    # smooth the data using the window and save to the new dict
                    try: 
                        self.baselines[session][data_type].append(baseline)
                    except: 
                        self.baselines[session][data_type] = [baseline]
                        
                    # build a linear regressor from the datapoints
                    linear_regressor = LinearRegression()
                    linear_regressor.fit(np.array(times[pre_start_i:start_i]).reshape(-1, 1) , np.array(values).reshape(-1, 1) )
                    
                    try: 
                        self.linreg[session][data_type].append(linear_regressor)
                    except:
                        self.linreg[session][data_type] = [linear_regressor]
                    
                    # Y_pred = linear_regressor.predict(X)  # make predictions
                    
                    # check that this isn't the first baseline in the list otherwise linear interpolation cannot happen
                    if self.baselines[session][data_type].index(baseline) != 0:
                        # use the baselines to build a line function
                        try:
                            self.lininterp[session][data_type].append(self.build_line_function(times[start_i]-(2*listening_session), times[start_i], self.baselines[session][data_type][-2], baseline))
                        except:
                            self.lininterp[session][data_type] = [self.build_line_function(times[start_i]-(2*listening_session), times[start_i], self.baselines[session][data_type][-2], baseline)]
    
    
    ''' method to crop the session data into the datapoints for individual songs '''
    def plot_song_data(self, timestamp_dict, session_no, song_no, data_type, buffer, plot_flatbaseline=False, plot_linreg=False, plot_lininterp=False):
        session_str = 'session_{}'.format(session_no)
        # use the session id and song number to extract the song information
        song = timestamp_dict[session_str].iloc[song_no-1]
        # find the start and end of the song playing
        start = int(np.ceil(song['timestamp_start']))
        end = int(np.ceil(song['timestamp_end']))
        # add seconds onto each side of the song on the plot to see the effect of the song
        start_plot = start - buffer
        end_plot = end + buffer
        
        # find the corresponding datapoint indexes to plot for a particular datatype
        times = self.data[session_str][data_type+'.csv']['timestamp']
        
        start_i = np.where(times<start_plot)[0][-1]
        end_i = np.where(times>end_plot)[0][0]
        
        song_start_i = np.where(times<start)[0][-1]
        song_end_i = np.where(times>end)[0][0]
        song_x = times[song_start_i:song_end_i]
        
        x = times[start_i:end_i]
        y = self.data_sm[session_str][data_type+'.csv'][start_i:end_i]
        
        # plot dat data
        with plt.style.context('seaborn-white'):  
            # instantiate figure
            plt.figure(figsize=(25, 5))
            plt.plot(x, y, label='actual values')
            plt.vlines(start, min(y), max(y), colors='m', linestyles='dashed', label='song start/end')
            plt.vlines(end, min(y), max(y), colors='m', linestyles='dashed')
            
            # plot the flat baseline
            if plot_flatbaseline:
                baseline = self.baselines[session_str][data_type+'.csv'][song_no-1]
                plt.hlines(baseline, start, end, color='c', label='flat baseline')
                
            # plot the linear regression baseline
            if plot_linreg:
                model = self.linreg[session_str][data_type+'.csv'][song_no-1]
                predictions = model.predict(np.array(song_x).reshape(-1,1))
                plt.plot(song_x, predictions, label='linear regression baseline')
            
            # plot the linear interpolation baseline
            if plot_lininterp:
                line = self.lininterp[session_str][data_type+'.csv'][song_no-1]
                plt.plot(song_x, line(song_x), label='linear interpolation baseline')
            
            plt.axis('tight')
            plt.title("Session {} Song {} {} data".format(session_no, song_no, data_type))
            plt.xlabel('Timestamp')
            plt.ylabel('{}'.format(data_type))
            plt.grid(True);
            plt.legend(loc="best")
            plt.show()
        
    
    ''' method to minus the effect of the walking baseline '''
    def relative_baseline(self, timestamp_dict, latency=3):
        # instantiate dictionary to save in the values after they have been standardized
        self.standardised_dict = {}
        # iterate through the different listening sessions
        
        for session in self.data_sm.keys():
            # create a new dictionary within the baseline dictionary for each session
            self.standardised_dict[session] = {}
            # create another dictionary for each datatype
            for data_type in self.data_sm[session].keys():
                self.standardised_dict[session][data_type] = {}
            
            # iterate through the songs 
            for song_no in range(timestamp_dict[session].shape[0]):
                song = timestamp_dict[session].iloc[song_no]
                # find the start time of the song playing
                start = int(np.ceil(song['timestamp_start']))
                end = int(np.ceil(song['timestamp_end']))
                # add the maximum latency to the start time
                start_l = start + latency
            
                # iterate through the different datafiles for each session:
                for data_type in self.data_sm[session].keys():
                    
                    # find the corresponding datapoint indexes to for a particular datatype
                    times = self.data[session][data_type]['timestamp']
                    start_l_i = np.where(times<start_l)[0][-1]
                    end_i = np.where(times>end)[0][0]
                    
                    # find the actual timestamps and datapoint values for the song
                    cropped_times = times[start_l_i:end_i]
                    values = self.data_sm[session][data_type][start_l_i:end_i]
                    
                    # find the predicted values from the flat, linear regression and linear interpolation baselines
                    flat_preds = np.repeat(self.baselines[session][data_type][song_no], len(values))
                    lr_preds = self.linreg[session][data_type][song_no].predict(np.array(cropped_times).reshape(-1,1))
                    try:
                        li_preds = self.lininterp[session][data_type][song_no](cropped_times)
                    # it will throw an index error on the last song, break out of the loop
                    except IndexError:
                        break
                    
                    # standardise the values by minusing the baseline predictions and averaging
                    vals_flat = (values - flat_preds).mean()
                    vals_lr = (values - lr_preds.reshape(-1)).mean()
                    vals_li = (values - li_preds).mean()
                    
                    # append to the dictionary
                    try:
                        self.standardised_dict[session][data_type]['flat baseline'].append(vals_flat)
                        self.standardised_dict[session][data_type]['lr baseline'].append(vals_lr)
                        self.standardised_dict[session][data_type]['li baseline'].append(vals_li)
                        
                    except:
                        self.standardised_dict[session][data_type]['flat baseline']=[vals_flat]
                        self.standardised_dict[session][data_type]['lr baseline']=[vals_lr]
                        self.standardised_dict[session][data_type]['li baseline']=[vals_li]
    
    
    ''' getter for the standardised dictionary '''
    def get_standardised_dict(self):
        return self.standardised_dict