import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# Database connection parameters
db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': '3306',  # default MySQL port is 3306
    'database': 'cdac_project'
}

# Create a SQLAlchemy engine using PyMySQL
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# Function to fetch unique tickers from the database
def fetch_tickers():
    tickers_query = "SELECT DISTINCT Ticker FROM nifty_data_002"
    tickers_df = pd.read_sql(tickers_query, engine)
    return tickers_df['Ticker'].tolist()

# Function to fetch data from Yahoo Finance and save to MySQL
def fetch_data_from_yahoo():
    # Define the tickers for Nifty 50 companies
    nifty_50_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ITC.NS", "BAJFINANCE.NS", "SBIN.NS", 
        "HCLTECH.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "HDFCLIFE.NS", "MARUTI.NS", "SUNPHARMA.NS", 
        "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS", "ADANIGREEN.NS", "TITAN.NS", "NESTLEIND.NS", 
        "POWERGRID.NS", "DIVISLAB.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "GRASIM.NS", 
        "TECHM.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "COALINDIA.NS", "HEROMOTOCO.NS", 
        "DRREDDY.NS", "BPCL.NS", "TATASTEEL.NS", "SHREECEM.NS", "CIPLA.NS", "BRITANNIA.NS", 
        "SBILIFE.NS", "ADANIENT.NS", "APOLLOHOSP.NS", "HINDALCO.NS", "TATACONSUM.NS", 
        "M&M.NS", "INDUSINDBK.NS", "UPL.NS", "BAJAJFINSV.NS", "VEDL.NS"
    ]
    # Fetch data for the last 5 years
    data_frames = []
    for stock in nifty_50_tickers:
        data = yf.download(stock, period='5y', interval='1d')
        data['Ticker'] = stock
        data.reset_index(inplace=True)
        data_frames.append(data)

    # Combine all data into a single DataFrame
    all_data = pd.concat(data_frames)
    all_data.rename(columns={'Date': 'Date_time', 'Adj Close': 'Adj_Close'}, inplace=True)
    st.write(all_data.head(), use_container_width=True)

    # Store the data in MySQL
    all_data.to_sql('nifty_data_002', con=engine, if_exists='replace', index=False)
    print("Data has been successfully stored in MySQL.")


# Function to fetch data for a selected ticker
def fetch_data(selected_ticker):
    query = f"SELECT Date_time, Close FROM nifty_data_002 WHERE Ticker = '{selected_ticker}' ORDER BY Date_time"
    data = pd.read_sql(query, engine)
    data['Date_time'] = pd.to_datetime(data['Date_time'])
    data.set_index('Date_time', inplace=True)
    return data

# Function to build and train the LSTM model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    return model

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Fetch Data", "Forecast", "About Us"])

if page == "Home":
    st.image("01.png" ,use_column_width=True)
    st.title("Deep Learning approach to mitigate financial risks")
    st.write("A LSTM model to forecast, analyze and categorize the stock levels into risk levels to mitigate the future financial risks.")

elif page == "Fetch Data":
    st.image("01.png" ,use_column_width=True)
    st.title("Fetch Data")
    
    if st.button("Fetch Data for Nifty 50"):
        fetch_data_from_yahoo()
        st.success("Data for all Nifty 50 stocks has been fetched and saved to the database.")

elif page == "Forecast":
    st.image("01.png" ,use_column_width=True)
    st.title("Forecast")
    tickers = fetch_tickers()
    selected_ticker = st.selectbox("Select a stock:", tickers)

    if st.button("Forecast"):
        # Fade out other elements
        st.markdown("<style>.fade { opacity: 0.3; }</style>", unsafe_allow_html=True)

        # Show the spinner while the model is running
        with st.spinner('Building model and forecasting the prices, please wait...'):
            data = fetch_data(selected_ticker)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Create sequences for LSTM
            def create_dataset(data, time_step=1):
                X, y = [], []
                for i in range(len(data) - time_step - 1):
                    X.append(data[i:(i + time_step), 0])
                    y.append(data[i + time_step, 0])
                return np.array(X), np.array(y)

            time_step = 60  # 60 days of historical data
            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = train_model(X_train, y_train)

            # Forecast the next 30 days
            last_data = scaled_data[-time_step:]
            forecast = []
            for _ in range(30):
                next_day = model.predict(last_data[np.newaxis, :, :])
                forecast.append(next_day[0, 0])
                last_data = np.concatenate([last_data[1:], [[next_day[0, 0]]]], axis=0)

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            # Create a DataFrame with forecasted prices and ticker name
            forecast_df = pd.DataFrame({
                'Date': pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30),
                'Forecasted_Price': forecast.flatten(),
                'Ticker': selected_ticker
            })

        # Clear the fade effect
        st.markdown("<style>.fade { opacity: 1; }</style>", unsafe_allow_html=True)

        # Display the forecasted DataFrame
        st.write("### Forecasted Prices for the Next 30 Days")
        st.dataframe(forecast_df, use_container_width=True)

        # Plotting the line chart
        st.line_chart(forecast_df, x='Date',y='Forecasted_Price', use_container_width=True)
        st.write("**Hover over the chart to see the forecasted prices.**")

        # Risk Analysis using Standard Deviation
        st.write("## Risk Analysis")
        mean_price = forecast_df['Forecasted_Price'].mean()
        std_dev = forecast_df['Forecasted_Price'].std()
        last_price = forecast_df['Forecasted_Price'].iloc[-1]

        if last_price > mean_price:
            risk_level = "Profit Making Stock"
            color = "green"
        elif last_price < mean_price - 2 * std_dev:
            risk_level = "High Risk Stock"
            color = "red"
        elif last_price < mean_price - std_dev:
            risk_level = "Moderate Risk Stock"
            color = "orange"
        elif last_price < mean_price:
            risk_level = "Low Risk Stock"
            color = "yellow"

        # Display risk level with color
        st.markdown(f"<h3 style='color:{color};'>Inference for {selected_ticker} : {risk_level}</h3>", unsafe_allow_html=True)
        st.image('risk matrix 101.png',width=500)

        # Download the forecast DataFrame as CSV
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_ticker}_forecast.csv", "text/csv")
      

elif page == "About Us":
    st.image("01.png" ,use_column_width=True)
    st.title("About Us")
    st.write("This project is developed by the team of CDAC PG-DBDA students as a part of final course completion project.")
    st.write("Meet our team:")

    # Team member information
    team_members = [
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Abhishek Nilekar'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Adesh Bakale'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Hrishikesh Shinde'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Abhinav Khade'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Ashlesha Lande'
        }
    ]

    # Display team members
    for member in team_members:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(member['image'], width=150)
        with col2:
            st.write(f"## {member['name']}")