
#  Data Modeling 
# 
# Ojective: Train and store optimized regression models for each currency pair, Build a new dictionary for parsing VOL and FD.
# 
# 1. Combine and clean 40 hours of the past currency datasets, and use 400 data points for model training. 
# 2. Loop through each currency pair, train and store models in the **models** folder.
# 3. Within the loop, build the dictionary for the trailing stops use. 


# Import required libraries
import datetime
import time
from polygon import RESTClient
from sqlalchemy import create_engine 
from sqlalchemy import text
import pandas as pd
from math import sqrt
from math import isnan
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from math import floor
import numpy as np
import pandas as pd
import pickle

# import pycaret
from pycaret.regression import *


currency_pairs = [["EUR","USD",[], []],
                  ["GBP","USD",[], []],
                  ["USD","CHF",[], []],
                  ["USD","CAD",[], []],
                  ["USD","HKD",[], []],
                  ["USD","AUD",[], []],
                  ["USD","NZD",[], []],
                  ["USD","SGD",[], []]]


# define a dictionary that we are going to use for storing sorted vol and fd parsing

dic_vol = {}
dic_fd = {}


# create engine to connect with 4 currency pairs database. 

engine1 = create_engine("sqlite+pysqlite:///../data/day1.db", echo=False, future=True)
engine2 = create_engine("sqlite+pysqlite:///../data/trailing.db", echo=False, future=True)
engine3 = create_engine("sqlite+pysqlite:///../data/day1_unclean.db", echo=False, future=True)
engine4 = create_engine("sqlite+pysqlite:///../data/day2_unclean.db", echo=False, future=True)


# Individual Sort Method

# create connections with 4 database, combine them and create regression models out of 40 hours of data.
with engine1.connect() as conn1:
    with engine2.connect() as conn2:
        with engine3.connect() as conn3:
            with engine4.connect() as conn4:
                for curr in currency_pairs:
                    df1 = pd.read_sql_table(curr[0]+curr[1]+"_agg",conn1)
                    df2 = pd.read_sql_table(curr[0]+curr[1]+"_agg",conn2)
                    df3 = pd.read_sql_table(curr[0]+curr[1]+"_agg",conn3)
                    df4 = pd.read_sql_table(curr[0]+curr[1]+"_agg",conn4)
                # clean up database1 and put in a dataframe
                    df1 = df1[['avgfxrate', 'vol','fd','return_r']]
                    df1 = df1.iloc[1:]
                    df1= df1.reset_index(drop=True)
                # clean up database2 and put in a dataframe
                    df2 = df2[['avgfxrate', 'vol','fd','return_r']]
                    df2 = df2.iloc[1:]
                    df2= df2.reset_index(drop=True) 
                # clean up database3 and put in a dataframe (VOL database3 is unconverted, and need to divide them by the mean)
                    df3 = df3[['avgfxrate', 'vol','fd','return_r']]
                    df3 = df3.iloc[1:]
                    df3= df3.reset_index(drop=True)
                    # change the VOL
                    df3['vol'] = df3['vol']/df3['avgfxrate']
                # clean up database4 and put in a dataframe (VOL database3 is unconverted, and need to divide them by the mean)
                    df4 = df4[['avgfxrate', 'vol','fd','return_r']]
                    df4 = df4.iloc[1:]
                    df4= df4.reset_index(drop=True)
                    # change the VOL
                    df4['vol'] = df4['vol']/df4['avgfxrate']
           
                # concatenate 4 database into our training dataframe
                    training = pd.concat([df1, df2,df3, df4], ignore_index=True)
                   
                # sort the traning set by vol and fd
                    training_vol = training.sort_values(by=['vol'], ascending=True)
                    training_vol.reset_index(drop=True, inplace=True)

                    training_fd = training.sort_values(by=['fd'], ascending=True)
                    training_fd.reset_index(drop=True, inplace=True)

                # init two varibale threshold 1 and 2 to note down the two breaking point, and put into the dictionary.
                    thres1_vol = training_vol['vol'][132]
                    thres2_vol = training_vol['vol'][264]

                    thres1_fd = training_fd['fd'][132]
                    thres2_fd = training_fd['fd'][264]
                
                # put threshold into dictionaries
                    dic_vol[curr[0]+curr[1]] = [thres1_vol, thres1_vol]
                    dic_fd[curr[0]+curr[1]] = [thres1_fd, thres2_fd]

                # define a function to put a new series to our training datasets
                    def parse_vol(series):
                        global thres1_vol
                        global thres2_vol
                        if series <= thres1_vol:
                            return 1
                        if series <= thres2_vol:
                            return 2
                        else:
                            return 3

                    def parse_fd(series):
                        global thres1_fd
                        global thres2_fd
                        if series <= thres1_fd:
                            return 1
                        if series <= thres2_fd:
                            return 2
                        else:
                            return 3

                    training['vol_rank'] = training['vol'].apply(parse_vol)
                    training['fd_rank'] = training['fd'].apply(parse_fd)

                # After coding to vol and fd in to [1,2,3], we need multiple the reurn by 100,000 to normalize the training output
                    training['return_r_label'] = training['return_r'].apply(lambda x: x * 100000)

                
                # Assign to our newly proccessed training dataset
                    training = training[['avgfxrate', 'vol_rank', 'fd_rank','return_r_label']]

                # we will init our categorical features and numeric features into a list
                    cate_features = ['vol_rank','fd_rank']
                    num_features = ['avgfxrate']

                # Now starting to model with Pycaret regression 
                    s = setup(data=training, target='return_r_label', 
                              categorical_features=cate_features, numeric_features=num_features,
                              verbose=False)
                # we have to exclude the following regression models because they don't generate predictions. 
                    best = compare_models(exclude=['llar','dummy','lasso', 'en'], verbose=False)
                    # evaluate_model(best)
                # save the model
                    save_model(best,f'../models/{curr[0]}{curr[1]}')
                    



# check and see our dictionary

with open ('../models/dic_vol.pkl', 'wb') as file:
    pickle.dump(dic_vol, file)

with open ('../models/dic_fd.pkl', 'wb') as file:
    pickle.dump(dic_fd, file)


