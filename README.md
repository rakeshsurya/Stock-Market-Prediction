# **Stock Market Prediction using LSTM in Python**

## **1. Introduction**

Stock price prediction is one of the most important and challenging tasks in finance. Predicting the future value of a stock is essential for investors who want to make profitable investments. The stock market is inherently unpredictable due to factors like economic trends, company performance, investor sentiment, and geopolitical events. Despite these challenges, advancements in machine learning and deep learning algorithms, such as Long Short-Term Memory (LSTM), have opened new opportunities to make predictions based on historical stock price data.

In this project, we use **LSTM**, a type of Recurrent Neural Network (RNN), to predict the future prices of a stock using past prices and technical indicators. We employ Python and several key libraries to create a robust solution that forecasts stock prices for up to 30 days into the future.

---

## **2. Purpose of the Project**

The main objective of this project is to:
- Build a machine learning model using **LSTM** to forecast stock prices based on historical stock data.
- Incorporate additional technical indicators like Moving Averages (MA), Relative Strength Index (RSI), and Bollinger Bands to improve the model’s performance.
- Evaluate the model’s predictive capability by comparing actual stock prices with the predicted values.

---

## **3. Libraries Used**

### 3.1 **yfinance**
- **Purpose**: This library allows easy access to historical stock data by interfacing with Yahoo Finance.
- **Usage in the Project**: We used `yfinance` to download historical stock prices for a given stock (e.g., Apple - AAPL) from 2010 to 2023.

### 3.2 **pandas**
- **Purpose**: Pandas is a powerful data manipulation library.
- **Usage in the Project**: We used pandas to handle the stock data, perform calculations like moving averages, and manage time series data.

### 3.3 **numpy**
- **Purpose**: Numpy provides support for large, multi-dimensional arrays and matrices.
- **Usage in the Project**: We used numpy to create sequences of data (for the LSTM input) and manage arrays of predictions.

### 3.4 **scikit-learn**
- **Purpose**: This library provides tools for machine learning, including data preprocessing and evaluation metrics.
- **Usage in the Project**: We used the `MinMaxScaler` from scikit-learn to normalize our stock data, which is essential for improving the performance of neural networks.

### 3.5 **tensorflow / keras**
- **Purpose**: TensorFlow is a deep learning library that includes Keras as a high-level API for building neural networks.
- **Usage in the Project**: We used Keras to build and train our LSTM model. The model includes LSTM layers for time series analysis, dropout layers to prevent overfitting, and dense layers for the final prediction.

### 3.6 **matplotlib**
- **Purpose**: Matplotlib is a plotting library for creating static, interactive, and animated visualizations in Python.
- **Usage in the Project**: We used matplotlib to visualize both the historical stock prices and the model’s predicted prices, making it easier to understand the performance of the model.

---

## **4. The Project Workflow**

### 4.1 **Step 1: Data Collection**
We use the `yfinance` library to download historical stock data for a given stock. The data includes the stock’s opening price, closing price, highest and lowest prices for the day, and trading volume.

```python
import yfinance as yf
stock_data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
```

### 4.2 **Step 2: Feature Engineering**
We added several features to the dataset, such as:
- **Moving Averages** (MA50 and MA200): These are used to smooth out price data and identify the trend direction.
- **RSI (Relative Strength Index)**: This is a momentum indicator that measures the speed and change of price movements.
- **Bollinger Bands**: These provide relative definitions of high and low prices.

```python
# Calculate 50-day and 200-day moving averages
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
```

### 4.3 **Step 3: Data Preprocessing**
Before training the LSTM model, we normalize the data using the `MinMaxScaler` to bring all features within the same range.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close', 'MA50', 'MA200', 'RSI', 'BB_upper', 'BB_lower']])
```

### 4.4 **Step 4: Build and Train the LSTM Model**
We build an LSTM model using the Keras API with two LSTM layers and dropout layers to prevent overfitting. The model is compiled using the Adam optimizer and trained on historical data.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

### 4.5 **Step 5: Forecasting Future Stock Prices**
We used the trained LSTM model to predict stock prices for the next 30 days. After making predictions, the results are scaled back to their original values using `MinMaxScaler.inverse_transform()`.

---

## **5. Analysis of the Graphs**

### **Graph 1: Predicted vs. Actual Stock Prices**

![Graph 1: Predicted vs. Actual Stock Prices](insert_image_path_here)

- **Analysis**: This graph shows the actual historical stock prices and the model’s predicted values for the test data. While the predicted values closely follow the actual prices, we see slight deviations. The LSTM model has captured the upward and downward trends reasonably well, indicating that the model can detect general patterns in the stock market. However, certain rapid price movements are harder to predict, as seen in some of the mismatches.

### **Graph 2: 30-Day Stock Price Forecast**

![Graph 2: 30-Day Stock Price Forecast](insert_image_path_here)

- **Analysis**: This graph represents the LSTM model’s forecast for the next 30 days of stock prices. The upward and downward trends forecasted by the model give a sense of future movement, though further analysis is required to evaluate long-term accuracy.

---

## **6. Conclusion**

In this project, we successfully built a stock price prediction model using LSTM, which is well-suited for time series data like stock prices. By incorporating features such as Moving Averages, RSI, and Bollinger Bands, we provided the model with additional context that helped improve prediction accuracy.

The results show that while the LSTM model is capable of capturing general price trends, it is less effective in predicting rapid fluctuations. Predicting stock prices is inherently difficult due to various unpredictable factors, and no model can guarantee perfect accuracy.

### **Why We Used These Libraries:**
- **yfinance** was chosen for easy access to stock data from Yahoo Finance.
- **pandas** allowed us to manipulate and analyze time series data.
- **numpy** provided the ability to efficiently handle arrays and mathematical operations.
- **scikit-learn** offered tools for normalizing data, which is crucial for deep learning.
- **tensorflow/keras** enabled us to build and train a robust LSTM model.
- **matplotlib** provided visualization capabilities to help us analyze the model’s performance.

This project serves as a demonstration of how machine learning and deep learning can be applied to stock market data for making future predictions, though it also highlights the limitations of any predictive model in such a volatile domain.
