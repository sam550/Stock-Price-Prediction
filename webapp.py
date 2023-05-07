import streamlit as st
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

## Stock Data Extraction using Merolagani website

# Define function to extract price history data
def extract_data(symbol):
    driver = webdriver.Chrome('C:\webdrivers\chromedriver.exe')
    driver.get(f'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}#0')
    driver.maximize_window()

    try:
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//a[@title='Price History']"))).click()
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
    return df_data

# Create input box for stock symbol
symbol = st.text_input('Enter stock symbol:')


# Create button to start data extraction
if st.button('Get Price History'):
    if symbol:
        # Call function to extract data
        df_data = extract_data(symbol)
        
        # Display extracted data as Pandas DataFrame
        st.write(df_data)
        
        df_data['Last_tp'] = df_data['Last_tp'].astype(str).str.replace(',', '').astype(float)

        df_data["Dates"] = pd.to_datetime(df_data["Dates"], format="%Y-%m-%d")
        df_data = df_data.set_index("Dates").sort_index()
        df_data["Last_tp"] = pd.to_numeric(df_data["Last_tp"])
        fig, ax = plt.subplots()
        ax.plot(df_data.index, df_data["Last_tp"])
        ax.set_title(f"{symbol} Stock Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        fig.savefig(f'{symbol}extracted_data_graph.png')
        st.pyplot(fig)
    else:
        st.warning('Please enter a stock symbol.')


# Train the data

if st.button('Train And Predict'):
    if symbol:
        
        fig = plt.imread(f'{symbol}extracted_data_graph.png')
        st.image(fig, use_column_width=True)
        df_data = pd.read_csv(f'{symbol} stock data.csv')
        #df_data = df_data.sort_values(by='Dates', ascending=False)
        st.write(df_data)
        
        df_data['Last_tp'] = df_data['Last_tp'].astype(str).str.replace(',', '').astype(float)

        st.write('Model is training...')
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

        st.write('Model trained successfully.')

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
        ax.set_xticks(range(0, len(df_data), 30))
        ax.set_xticklabels(df_data['Dates'][::30],rotation=45)
        ax.set_xlim(0, len(df_data))

        #plt.xticks(rotation=45)
        st.pyplot(fig)


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
        st.markdown("<h2 style='text-align:center;font-weight:bold;'>Predicted Graph</h2>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(16,8))
        ax.plot(df_data['Last_tp'])
        ax.plot(np.arange(len(df_data['Last_tp']), len(df_data['Last_tp']) + len(future_predictions)), future_predictions)
        ax.set_title('Predicted Stock Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend(['Actual', 'Predicted'])
        ax.set_xticklabels(df_data['Dates'])
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(future_predictions)
        validation_actual = validation['Last_tp'].values
        validation_predictions = validation['Predictions'].values
        mape = np.mean(np.abs((validation_actual - validation_predictions) / validation_actual)) * 100
        st.write('MAPE:', mape)
    else:
        st.write('Generate the data first!!!')
  
