import flask 
from flask import Flask ,request,render_template
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
import json


app=Flask(__name__)
@app.route('/extract_data/<symbol>', methods=['GET'])

def extract_data(symbol):
    driver = webdriver.Chrome('C:\webdrivers\chromedriver.exe')
    driver.get(f'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}#0')
    driver.maximize_window()

    try:
        WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.XPATH, "//a[@title='Price History']"))).click()
    except:
        pass

    result = []
    max_pages = 2
    page = 1
    while page <= max_pages:
        try:
            dates = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//table[@class='table table-bordered table-striped table-hover']/tbody/tr/td[2]")))
            last_tp = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, "//table[@class='table table-bordered table-striped table-hover']/tbody/tr/td[3]")))
            change = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, "//table[@class='table table-bordered table-striped table-hover']/tbody/tr/td[4]")))
            high = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, "//table[@class='table table-bordered table-striped table-hover']/tbody/tr/td[5]")))
            low = WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, "//table[@class='table table-bordered table-striped table-hover']/tbody/tr/td[6]")))

            for i in range(len(dates)):
                data ={'Dates': dates[i].text,'Last_tp':last_tp[i].text,'Change':change[i].text,'High':high[i].text,'Low':low[i].text}
                result.append(data)

            next_elements = driver.find_elements(By.XPATH, "//a[@title ='Next Page']")
            if next_elements:
                next_elements[0].click()
                page +=1
            else:
                break
        except:
            break

    driver.quit()
    df_data = pd.DataFrame(result)
    df_data = df_data.sort_values(by='Dates', ascending=True)
    df_data = df_data.reset_index(drop=True)
    df_data.index += 1
    df_data.to_csv(f'{symbol} stock data.csv', index=False)
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(df_data['Dates'], df_data['Last_tp'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.grid()
    ax.set_title('Plotted Stock Price', fontweight='bold', loc='center')
    ax.set_xticklabels(df_data['Dates'])
    plt.xticks(rotation=45)
    fig.savefig(f'{symbol}extracted_data_graph.png')
   

    result = df_data.to_json(orient="split") 
    return result


#creating route 
@app.route("/") 
def index():
    
    return render_template("index.html")

@app.route('/train_data/<symbol>', methods=['GET'])
def train_data(symbol):
        # fig = plt.imread(f'extracted_data_graph.png')
        df_data = pd.read_csv(f'{symbol} stock data.csv')
        close_prices = df_data['Last_tp']
        values = close_prices.values
        training_data_len = math.ceil(len(values)* 0.8)

        # scale data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))

        # split data into training and testing sets
        train_data = scaled_data[0: training_data_len, :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_data[training_data_len-60: , : ]
        x_test = []
        y_test = values[training_data_len:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # build and train model
        model = keras.Sequential()
        model.add(layers.GRU(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(100, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=10)

    # To predict the data
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)

        data = df_data.filter(['Last_tp'])
        train = data[:training_data_len]
        validation = data[training_data_len:]
        validation['Predictions'] = predictions
        
        fig,ax = plt.subplots(figsize=(16,8))
        ax.set_title('Model')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.plot(train)
        ax.plot(validation[['Last_tp', 'Predictions']])
        ax.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        ax.set_xticklabels(df_data['Dates'])
        plt.xticks(rotation=45)

        # Get the last 60 days of closing prices
        last_60_days = df_data['Last_tp'][-60:].values.reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        future_predictions = []
        for i in range(30):
            X_test = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))
            y_pred = model.predict(X_test)
            y_pred_actual = scaler.inverse_transform(y_pred)
            future_predictions.append(y_pred_actual[0][0])
            last_60_days_scaled = np.vstack([last_60_days_scaled[1:], y_pred])

        # Plot the predicted values
        
        fig,ax = plt.subplots(figsize=(16,8))
        ax.plot(df_data['Last_tp'])
        ax.plot(np.arange(len(df_data['Last_tp']), len(df_data['Last_tp']) + len(future_predictions)), future_predictions)
        ax.set_title('Predicted Stock Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend(['Actual', 'Predicted'])
        ax.set_xticklabels(df_data['Dates'])
        plt.xticks(rotation=45)
        fig.savefig(f'{symbol}future_data_graph.png')
        future_predictions = np.array(future_predictions)
        my_list = future_predictions.astype(float).tolist()
        with open(f'{symbol}predicted.json', "w") as f:
            # Write the JSON data to the file
            json.dump(my_list, f)

        json_string = json.dumps(my_list)
        return json_string
       
        
if __name__=="__main__":
    app.run(port=8085) 