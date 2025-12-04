import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import load_data, scale_data, create_dataset
from src.model import build_lstm_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 1) Veriyi yükle
data = load_data("data/goldprice.csv")
prices = data['Gold_Price'].values.reshape(-1, 1)

# 2) Ölçekleme
scaled_prices, scaler = scale_data(prices)

# 3) Dataset hazırla
time_step = 12
X, y = create_dataset(scaled_prices, time_step)

# 4) Train-test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 5) Modeli oluştur
model = build_lstm_model((X_train.shape[1], 1))

# 6) Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=5)

# 7) Modeli eğit
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)

# 8) Gelecek 6 ay için tahmin
last_12 = scaled_prices[-time_step:].reshape(1, time_step, 1)
predictions = []

for _ in range(6):
    pred = model.predict(last_12)
    predictions.append(pred[0][0])
    last_12 = np.append(last_12[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
future_months = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=6, freq='M')

# 9) Grafik
plt.figure(figsize=(12, 6))
plt.plot(data.index, prices, label='Gerçek Altın Fiyatları', color='blue')
plt.plot(future_months, predicted_prices, label='Tahmin Edilen Altın Fiyatları', color='red')
plt.title('Altın Fiyatları Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Altın Fiyatı')
plt.legend()
plt.savefig("results/prediction.png")
plt.show()

print("\nÖnümüzdeki 6 ay için tahmin edilen altın fiyatları:")
for month, price in zip(future_months, predicted_prices):
    print(f"{month.strftime('%Y-%m')}: {price[0]:.2f} TL")
