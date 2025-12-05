# Gold Price Prediction with LSTM

A deep learning project that predicts future gold prices using Long Short-Term Memory (LSTM) neural networks.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📊 Results

![Gold Price Prediction](results/prediction.png)

### Model Performance

| Metric | Value |
|--------|-------|
| Model | 2-Layer LSTM (150-150 neurons) |
| Final Training Loss (MSE) | 0.0022 |
| Time Step | 12 months |
| Forecast Period | 6 months |
| Training Epochs | 15 (Early Stopping) |

### 🔮 6-Month Price Forecast (TL)

| Month | Predicted Price |
|-------|-----------------|
| November 2024 | 2,946.88 TL |
| December 2024 | 3,063.03 TL |
| January 2025 | 3,176.24 TL |
| February 2025 | 3,286.90 TL |
| March 2025 | 3,398.92 TL |
| April 2025 | 3,502.94 TL |

---

## 🏗️ Project Structure

    gold-price-prediction/
    ├── data/
    │   └── goldprice.csv
    ├── notebooks/
    │   └── GoldPredict.ipynb
    ├── src/
    │   ├── model.py
    │   ├── preprocessing.py
    │   └── train.py
    ├── results/
    │   └── prediction.png
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## 📈 Dataset

| Info | Details |
|------|---------|
| **Source** | Manually collected |
| **Period** | January 2020 - October 2024 |
| **Records** | 58 monthly data points |
| **Feature** | Monthly average gold prices (TL) |
| **Start Price** | 297.01 TL |
| **End Price** | 3,081.60 TL |

---

## 🧠 Model Architecture

    Input (12 months)
           ↓
    ┌─────────────────────────┐
    │ LSTM Layer 1            │
    │ 150 units               │
    │ return_sequences=True   │
    └─────────────────────────┘
           ↓
    ┌─────────────────────────┐
    │ Dropout (30%)           │
    └─────────────────────────┘
           ↓
    ┌─────────────────────────┐
    │ LSTM Layer 2            │
    │ 150 units               │
    └─────────────────────────┘
           ↓
    ┌─────────────────────────┐
    │ Dropout (30%)           │
    └─────────────────────────┘
           ↓
    ┌─────────────────────────┐
    │ Dense Layer (1)         │
    │ Output: Next Price      │
    └─────────────────────────┘

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Max Epochs | 100 |
| Early Stopping | Patience: 5 |
| Batch Size | 32 |
| Train/Test Split | 80% / 20% |

---

## 🚀 Getting Started

### 1. Clone the Repository

    git clone https://github.com/Xiast-sw/gold-price-prediction.git
    cd gold-price-prediction

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Run Training

    python -m src.train

### 4. Or Use Jupyter Notebook

    jupyter notebook notebooks/GoldPredict.ipynb

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **ML Tools** | Scikit-learn |

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| src/model.py | LSTM model architecture definition |
| src/preprocessing.py | Data loading, scaling, and dataset creation |
| src/train.py | Main script to train model and generate predictions |
| notebooks/GoldPredict.ipynb | Interactive analysis and experimentation |
| data/goldprice.csv | Historical gold price data |

---

## 👤 Author

**Adil Buğra Aytar**

[![GitHub](https://img.shields.io/badge/GitHub-Xiast--sw-black?logo=github)](https://github.com/Xiast-sw)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adil%20Buğra%20Aytar-blue?logo=linkedin)](https://linkedin.com/in/adil-bugra-aytar-47a555224)
[![Email](https://img.shields.io/badge/Email-a.bugraaytar@gmail.com-red?logo=gmail)](mailto:a.bugraaytar@gmail.com)

---

## 📝 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, please consider giving it a star!
