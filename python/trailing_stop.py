#  Build an optimized real-time trailing-stop-strategy, and use our model prediction to make real-time investment decisions.
 
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
from get_data import *
# import pycaret
from pycaret.regression import *

# This is the values that we got by sorting vol and fd for modeling 
with open('../models/dic_vol.pkl', 'rb') as file:
    dic_vol = pickle.load(file)

with open('../models/dic_fd.pkl', 'rb') as file:
    dic_fd = pickle.load(file)


# Create a dictionary to record the previous error 
previous_error = {}

 


 
#  Modeling Tables

# We will create a table for our modeling prediction. (The table is running every 6 minutes and predict the return values from the agg table)

# MLResult Table:
#  Predicted return, Actual Return, Error

# This creates model output tables with the attributes of predicted return, the actual return, and the error.
def initialize_model_output_tables(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs:
            conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_MLResult(predicted_return numeric, actual_return numeric, error numeric);"))

# This function will execute our models that were predicted from data_engineering_modeling.ipynb, and insert our result into our modeling tables.
def aggregate_agg_data_to_ML(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs:
            df = pd.read_sql_table(curr[0]+curr[1]+"_agg",conn)
            # Check and see if it's the first entry. If it's true, there is no return.
            if df['avgfxrate'].count() == 1:
                predicted_return = None
                actual_return = None
                error = None 
                # insert value into our database
                conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                              '''_MLResult VALUES (:predicted_return, :actual_return, :error );'''),
                         {'predicted_return':predicted_return, 'actual_return':actual_return, 'error':error})
            else:
                training = df[['avgfxrate', 'vol','fd','return_r']].iloc[-1:]
                training_copy = training.copy()

                # Build a recursive function to run in parse_vol_fd function in order to 
                # access our dictionary in to begining, and define the new threshold for parsing fd and vol into [1,2,3]
                def get_thres_vol(curr):
                    global dic_vol
                    thres1 = dic_vol[curr[0]+curr[1]][0]
                    thres2 = dic_vol[curr[0]+curr[1]][1]
                    return [thres1,thres2]

                def get_thres_fd(curr):
                    global dic_fd
                    thres1 = dic_fd[curr[0]+curr[1]][0]
                    thres2 = dic_fd[curr[0]+curr[1]][1]
                    return [thres1,thres2]

                # define a function to put a new series to our training dataframe
                def parse_vol(series, curr):
                    thres1 = get_thres_vol(curr)[0]
                    thres2 = get_thres_vol(curr)[1]
                    if series <= thres1:
                        return 1
                    if series <= thres2:
                        return 2
                    else:
                        return 3

                def parse_fd(series, curr):
                    thres1 = get_thres_fd(curr)[0]
                    thres2 = get_thres_fd(curr)[1]
                    if series <= thres1:
                        return 1
                    if series <= thres2:
                        return 2
                    else:
                        return 3
                
                training['vol_rank'] = training['vol'].apply(parse_vol, args=(curr,))
                training['fd_rank'] = training['fd'].apply(parse_fd, args=(curr,))


                # Get our training dataframe ready for prediction
                training = training[['avgfxrate', 'vol_rank', 'fd_rank']]
                #load our model
                model = load_model(f'../models/{curr[0]}{curr[1]}')
                # make prediction 
                prediction = predict_model(model, data=training)
                # print(prediction)

                predicted_return = float(prediction['prediction_label'].values)
                # we need to divide the predicted value by 100,000 to get the actual prediction
                predicted_return = predicted_return/100000
                actual_return = float(training_copy['return_r'].values)
                error = predicted_return - actual_return

                # insert value into our database
                conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                              '''_MLResult VALUES (:predicted_return, :actual_return, :error );'''),
                         {'predicted_return':predicted_return, 'actual_return':actual_return, 'error':error})

# This is the trailing stops tables session. We have go long and go short strategys. In here, I choose 4 currency pairs go long and 4 currency pairs go short. 
# In each hour, we will make investment decision base on our model predictions, our previous errors, and our actual return. 

# Long Tables:
#  Balance, Profit_Loss, Status

# Short Tables:
#  Balance, Profit_Loss, Status

# create tables for trailing stops project
def initialize_trailing_stops_data_tables(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs[:4]:
            conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_long(balance numeric, profit_loss numeric, status text);"))
        for curr in currency_pairs[4:]:
            conn.execute(text("CREATE TABLE "+curr[0]+curr[1]+"_short(balance numeric, profit_loss numeric, status text);"))

# The follow two function will aggregate the data for trailing tables
# This function will fill values in the go long tables.
def agg_for_trailing_stops_data_long_tables(engine,currency_pairs):
    with engine.begin() as conn:
        for curr in currency_pairs[:4]:
            # Init layer
            layer = {1: -0.0025, 2: -0.0015, 3: -0.001, 4: -0.0005, 5: -0.0005, 6: -0.0005,7 : -0.0005,
                    8: -0.0005, 9: -0.0005, 10: -0.0005}
            # check which layer that we are in
            count_row = conn.execute(text("SELECT count(*) AS count_ FROM "+curr[0]+curr[1]+"_long;"))
            for row in count_row:
                count_ = row.count_
            layer_we_in = count_ + 1
            
            # first, I am going to the MLResult tables to get the actual return values and calcate the sum of last 10 values
            df = pd.read_sql_table(curr[0]+curr[1]+"_MLResult",conn)
            sum_of_actual_return = df['actual_return'][-10:].sum()

            # second, sum the current predicted_return. (current)
            sum_of_predicted_return = df['predicted_return'][-10:].sum()
            
            # init and check the balance
            if count_ == 0:
                balance = 100
            else:
                balance = conn.execute(text("SELECT balance FROM "+curr[0]+curr[1]+"_long;"))
                balance = [row.balance for row in balance][-1]
            
            # check if the stauts were closed, if ture break out of the current loop, jump to the next.
            if count_ >= 1:
                curr_status = conn.execute(text("SELECT status FROM "+curr[0]+curr[1]+"_long;"))
                curr_status = [row.status for row in curr_status][-1]
                if curr_status == 'close':
                    continue               
                 
            # use conditions statement to compare layer values with sum_of_actual_return
            status = 'continue'
            if sum_of_actual_return > layer[layer_we_in]:
                if layer_we_in <= 4:
                    # First to check whether the (current actual returns) are greater or smaller than O to see if we are losing or gaining money.
                    # then compare the (current predictions with the privous errors) with (current actual returns) to see what action to take.
                    if sum_of_actual_return > 0:
                        try:
                            if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) > sum_of_actual_return:
                                profit = balance * sum_of_actual_return
                                balance = profit + balance + 100
                            else:
                                profit = balance * sum_of_actual_return
                                balance = profit + balance
                        except:
                            profit = balance * sum_of_actual_return
                            balance = profit + balance + 100                 
                    else:
                        try:
                            if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) > sum_of_actual_return:
                                profit = balance * sum_of_actual_return
                                balance = profit + balance
                            else:
                                profit = balance * sum_of_actual_return
                                balance = profit + balance
                                status = 'close'
                        except:
                            profit = balance * sum_of_actual_return
                            balance = profit + balance + 100
                    # insert values into the table
                    conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                      '''_long VALUES (:balance, :profit_loss, :status);'''),
                                 {'balance': balance, 'profit_loss': profit,  'status': status})
                else:
                    if sum_of_actual_return > 0:     
                        profit = balance * sum_of_actual_return
                        balance = profit + balance                         
                    else:  
                        if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) > sum_of_actual_return:
                            profit = balance * sum_of_actual_return
                            balance = profit + balance
                        else:
                            profit = balance * sum_of_actual_return
                            balance = profit + balance
                            status = 'close'

                    #insert the values into the tables
                    conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                      '''_long VALUES (:balance, :profit_loss, :status);'''),
                                 {'balance': balance, 'profit_loss': profit,  'status': 'continue'})
            else:
                profit = balance * sum_of_actual_return
                balance = profit + balance
                status = 'close'
                #insert the values into the tables
                conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                  '''_long VALUES (:balance, :profit_loss, :status);'''),
                             {'balance': balance, 'profit_loss': profit,  'status': status}) 

            # finally, do the sum of error (we want the privous error), so we will compute to current error and store it in the previous error dictionary for the next use.
            sum_of_previous_error = df['error'][-10:].sum()
            previous_error[curr[0]+curr[1]] = sum_of_previous_error

# This function will fill values in the go short tables.
def agg_for_trailing_stops_data_short_tables(engine,currency_pairs):
    with engine.begin() as conn:        
        for curr in currency_pairs[4:]:
            # Init layer
            layer = {1: 0.0025, 2: 0.0015, 3: 0.001, 4: 0.0005, 5: 0.0005, 6: 0.0005,7 : 0.0005,
                    8: 0.0005, 9: 0.0005, 10: 0.0005}
            # check which layer that we are in
            count_row = conn.execute(text("SELECT count(*) AS count_ FROM "+curr[0]+curr[1]+"_short;"))
            for row in count_row:
                count_ = row.count_
            layer_we_in = count_ + 1
            
            # first, I am going to the MLResult tables to get the actual return values and calcate the sum of last 10 values
            df = pd.read_sql_table(curr[0]+curr[1]+"_MLResult",conn)
            sum_of_actual_return = df['actual_return'][-10:].sum()

            # second, sum the current predicted_return. (current)
            sum_of_predicted_return = df['predicted_return'][-10:].sum()
            
            # init and check the balance
            if count_ == 0:
                balance = 100
            else:
                balance = conn.execute(text("SELECT balance FROM "+curr[0]+curr[1]+"_short;"))
                balance = [row.balance for row in balance][-1]

            # check if the stauts were closed, if ture break out of the current loop, jump to the next.
            if count_ >= 1:
                curr_status = conn.execute(text("SELECT status FROM "+curr[0]+curr[1]+"_short;"))
                curr_status = [row.status for row in curr_status][-1]
                if curr_status == 'close':
                    continue

            # use conditions statement to compare layer values with sum_of_actual_return.
            status = 'continue'
            if sum_of_actual_return < layer[layer_we_in]:
                if layer_we_in <= 4: 
                    # First to check whether the (current actual returns) are greater or smaller than O to see if we are losing or gaining money.
                    # then compare the (current predictions with the privous errors) with (current actual returns) to see what action to take.
                    if sum_of_actual_return < 0:
                        try:
                            if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) < sum_of_actual_return:
                                profit = balance * sum_of_actual_return* (-1)
                                balance = profit + balance + 100
                            else:
                                profit = balance * sum_of_actual_return* (-1)
                                balance = profit + balance
                        except:
                            profit = balance * sum_of_actual_return* (-1)
                            balance = profit + balance + 100                 
                    else:
                        try:
                            if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) < sum_of_actual_return:
                                profit = balance * sum_of_actual_return* (-1)
                                balance = profit + balance
                            else:
                                profit = balance * sum_of_actual_return* (-1)
                                balance = profit + balance
                                status = 'close'
                        except:
                            profit = balance * sum_of_actual_return* (-1)
                            balance = profit + balance + 100
                    #insert the values into the tables
                    conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                      '''_short VALUES (:balance, :profit_loss, :status);'''),
                                 {'balance': balance, 'profit_loss': profit,  'status': status})
                else:
                    if sum_of_actual_return < 0:     
                        profit = balance * sum_of_actual_return* (-1)
                        balance = profit + balance                         
                    else:  
                        if (sum_of_predicted_return - previous_error[curr[0]+curr[1]]) < sum_of_actual_return:
                            profit = balance * sum_of_actual_return* (-1)
                            balance = profit + balance
                        else:
                            profit = balance * sum_of_actual_return* (-1)
                            balance = profit + balance
                            status = 'close'
                    #insert the values into the tables
                    conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                      '''_short VALUES (:balance, :profit_loss, :status);'''),
                                 {'balance': balance, 'profit_loss': profit,  'status': status})
            else:
                profit = balance * sum_of_actual_return * (-1)
                balance = profit + balance
                status = 'close'
                #insert the values into the tables
                conn.execute(text("INSERT INTO "+curr[0]+curr[1]+
                                  '''_short VALUES (:balance, :profit_loss, :status);'''),
                             {'balance': balance, 'profit_loss': profit,  'status': status})  
            
            # finally, do the sum of error (we want the privous error), so we will compute to current error and store it in the previous error dictionary for the next use.
            sum_of_previous_error = df['error'][-10:].sum()
            previous_error[curr[0]+curr[1]] = sum_of_previous_error
            

#  Main Function 

# Our main function will execute the previous tables in order to generate outputs.

# This main function repeatedly calls the polygon api every 1 seconds for 10 hours 
# and stores the results.
def main(currency_pairs):
    # The api key given by the professor
    key = "beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq"
   
    # Number of list iterations - each one should last about 1 second
    count = 0
    agg_count = 0
    ts_count = 0
    times_count = 0
    
    # Create an engine to connect to the database; setting echo to false should stop it from logging in std.out
    engine = create_engine("sqlite+pysqlite:///../data/test.db", echo=False, future=True)
    
    # Create the needed tables in the database
    initialize_raw_data_tables(engine,currency_pairs)
    initialize_aggregated_tables(engine,currency_pairs)
    initialize_model_output_tables(engine,currency_pairs)
    initialize_trailing_stops_data_tables(engine,currency_pairs)
    
    # Open a RESTClient for making the api calls
    client = RESTClient(key)
    # Loop that runs until the total duration of the program hits 10 hours. 
    while count <= 36000: # 36000 seconds = 10 hours 
        
        # Make a check to see if 6 minutes has been reached or not 
        if agg_count == 360:
            # Aggregate the data and clear the raw data tables
            aggregate_raw_data_tables(engine,currency_pairs)
            reset_raw_data_tables(engine,currency_pairs)
            # put agg value into a model prediction
            aggregate_agg_data_to_ML(engine,currency_pairs)
            agg_count = 0
            times_count += 1
            print(f"finish {times_count} times prediction!")
        
        # check if one hour has been reached 
        if ts_count == 3600:
            # call function and aggreate for the trailing stops tables
            agg_for_trailing_stops_data_long_tables(engine,currency_pairs)
            agg_for_trailing_stops_data_short_tables(engine,currency_pairs)
            ts_count = 0

        # Only call the api every 1 second, so wait here for 0.75 seconds, because the 
        # code takes about .15 seconds to run
        time.sleep(0.75)

        # Increment the counters
        count += 1
        agg_count +=1
        ts_count += 1

        # Loop through each currency pair
        for currency in currency_pairs:
            # Set the input variables to the API
            from_ = currency[0]
            to = currency[1]

            # Call the API with the required parameters
            try:
                resp = client.get_real_time_currency_conversion(from_, to, amount=100, precision=2)
            except:
                continue

            # This gets the Last Trade object defined in the API Resource
            last_trade = resp.last

            # Format the timestamp from the result
            dt = ts_to_datetime(last_trade.timestamp)

            # Get the current time and format it
            insert_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Calculate the price by taking the average of the bid and ask prices
            avg_price = (last_trade.bid + last_trade.ask)/2

            # Write the data to the SQLite database, raw data tables
            with engine.begin() as conn:
                conn.execute(text("INSERT INTO "+from_+to+"_raw(ticktime, fxrate, inserttime) VALUES (:ticktime, :fxrate, :inserttime)"),[{"ticktime": dt, "fxrate": avg_price, "inserttime": insert_time}])



# A dictionary defining the set of currency pairs we will be pulling data for
currency_pairs = [["EUR","USD",[], []],
                  ["GBP","USD",[], []],
                  ["USD","CHF",[], []],
                  ["USD","CAD",[], []],
                  ["USD","HKD",[], []],
                  ["USD","AUD",[], []],
                  ["USD","NZD",[], []],
                  ["USD","SGD",[], []]]

# Run the main data collection loop
if __name__ == '__main__':
    main(currency_pairs)

    # save the long and short tables to csv
    engine = create_engine("sqlite+pysqlite:///../data/test.db", echo=False, future=True)
    with engine.connect() as conn:
        for curr in currency_pairs[:4]:
            table = conn.execute(text("SELECT * FROM "+curr[0]+curr[1]+"_long;"))
            table = table.fetchall()
            df = pd.DataFrame(table)
            df.to_csv(f'../output_csv/{curr[0]}{curr[1]}_long.csv' ,index=False)
        for curr in currency_pairs[4:]:
            table = conn.execute(text("SELECT * FROM "+curr[0]+curr[1]+"_short;"))
            table = table.fetchall()
            df = pd.DataFrame(table)
            df.to_csv(f'../output_csv/{curr[0]}{curr[1]}_short.csv' ,index=False)

