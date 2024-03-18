import pickle
from django.shortcuts import render
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas_datareader as web
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.models import load_model
import keras


def home(request):
    return render(request, 'index.html')


def smart_forcasting(request, sub):
    if request.method == 'POST':
        datas = request.FILES['datas']
        df = pd.read_csv(datas)
        new_df = df.filter([sub])

        dataset = new_df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        pd.DataFrame(scaled_data)


        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # loading the trained model
        if sub == 'Water ':
            model = pickle.load(open(
                'C:/Users/syamp/PycharmProjects/MeterForcasting/smart_meter/expenses/models/Water_Price_forecasting.h5',
                "rb"))
        elif sub == 'Gas':
            model = pickle.load(open(
                'C:/Users/syamp/PycharmProjects/MeterForcasting/smart_meter/expenses/models/Gas_forecasting_A.h5',
                "rb"))
        elif sub == 'Electricity':
            model = pickle.load(open(
                'C:/Users/syamp/PycharmProjects/MeterForcasting/smart_meter/expenses/models/ELectricity_forecasing.h5',
                "rb"))
        else:
            message = "wrong input"
            return render(request, 'index.html',{' message ':  message })


        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        price = pred_price[0][0]
    return render(request, 'index.html',{'price': price})


