# Deep-Learning-approach-to-mitigate-financial-risk

A deep learning model to forecast, analize and Categorize the future prices of nifty 50 stocks to mitigate future financian risks.


### Project Architecture : 
![Slide2](https://github.com/user-attachments/assets/0471166e-afec-4682-852b-a57d8824d350)

### Execution flow : 
1. Data Extraction : Extract the data from Yahoo Finance API for last 5 years daily data using Pandas library. The Fetch data activity can be triggered through the streamlit UI as well.
2. Data Transformation : performed using Pandas as data is already clean, we just changed the column names of Date column and Adjusted close price column so avoid any future errors while saving the data in database.
3. Data Loading : Load the extracted data into MySQL database to better manage the data so that it can be used to fetch the data in future model trainig as well
4. Model training : The model will be trained as the user will select the desired stock from the UI, as a result the model will predict the prices of next 30 days and It will infer the risk level of that stock based on standard deviation of forecasted data.
5. Streamlit UI : all the above activities can be done using Streamlit UI where user can directly fetch the new data and perform forecast of his desired stock.
6. Model forecast : the forecasted prices will be displayed on UI
7. Forecast analysis and Visualization : the complete forecast analysis of all the 50 Nifty stocks are done using Tableau and Power BI reports.

### Comparative analysis with ARIMA and SARIMAX models : 
![arima](https://github.com/user-attachments/assets/f0c4962f-6287-47dd-94b0-76d4f5319163)
  ![sarimax](https://github.com/user-attachments/assets/82c78b80-624a-402a-994d-ac610418c23b)

### Model Visualization : 
![viz my_lstm_model h5](https://github.com/user-attachments/assets/2db8af45-e3f5-493e-add3-f42c184689fe)

### Model Summary : 
![Screenshot 2024-08-15 153214](https://github.com/user-attachments/assets/56a39415-afa3-4e4f-8ae9-c37c52ced7fc)

### Accuracy of the model : 
![accuracy score](https://github.com/user-attachments/assets/8c94c261-9bc7-43b3-85b7-5e45286276df)
