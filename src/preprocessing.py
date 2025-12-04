import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    """CSV dosyasını yükler ve tarih indeksini ayarlar."""
    data = pd.read_csv(path, sep=';', decimal=',')
    data['Month'] = pd.to_datetime(data['Month'], format='%m-%Y')
    data.set_index('Month', inplace=True)
    return data

def scale_data(values):
    """Veriyi MinMaxScaler ile ölçekler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled, scaler

def create_dataset(data, time_step=12):
    """LSTM giriş formatına uygun veri üretir."""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)
