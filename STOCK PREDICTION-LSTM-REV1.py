#!/usr/bin/env python
# coding: utf-8

# ### PACKAGES

# In[ ]:


pip install nsetools


# In[ ]:


pip install yfinance


# In[ ]:


pip install yfinance pandas mplfinance TA-Lib


# In[115]:


pip install tensorflow==2.12.0


# In[118]:


pip install keras


# In[30]:


pip install fuzzywuzzy


# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fuzzywuzzy import process


# ### Getting the Data

# In[2]:


#IMPORTING ALL EQUITY SYMBOLS
tickers_all=pd.read_csv("D:\DSML PG\CAPSTONE PROJECT\EQUITY_L.csv")
tickers_all.head()


# In[5]:


# Function to preprocess company names and perform fuzzy matching
def get_symbol(company_name, tickers_all):
    # Preprocess company names to lowercase
    tickers_all['LOWER_COMPANY_NAME'] = tickers_all['NAME OF COMPANY'].str.lower()
    
    # Perform fuzzy matching to get nearby suggestions
    matches = process.extract(company_name.lower(), tickers_all['LOWER_COMPANY_NAME'], limit=5)
    
    # Print nearby suggestions
    print("Nearby Suggestions:")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. {match[0]}")

    # Ask the user to select one of the suggestions
    selection = int(input("Enter the number corresponding to your selection (or 0 to enter manually): "))
    if selection == 0:
        selected_company_name = input("Enter the company name manually: ")
        company_data = tickers_all[tickers_all["LOWER_COMPANY_NAME"] == selected_company_name.lower()]
    elif 1 <= selection <= len(matches):
        selected_company_name = matches[selection - 1][0]
        company_data = tickers_all[tickers_all["LOWER_COMPANY_NAME"] == selected_company_name.lower()]
    else:
        print("Invalid selection.")
        return None

    # Extract the symbol for the selected company
    if not company_data.empty:
        symbol = company_data.iloc[0]["SYMBOL"]
        symbol_with_extension = symbol + ".NS"  # Add .NS to the symbol
        print(f"Symbol for {selected_company_name}: {symbol_with_extension}")
        return {"ticker": symbol_with_extension, "company_name": selected_company_name}
    else:
        print(f"No data found for {selected_company_name}")
        return None

# INPUT REQUIRED COMPANY NAME
company_name = input('Enter the company name: ')
result = get_symbol(company_name, tickers_all)
if result:
    ticker = result["ticker"]
    selected_company_name = result["company_name"]
    print("Symbol saved in ticker:", ticker)
    print("Company name saved:", selected_company_name)
else:
    print("No symbol found.")


# In[6]:


ticker


# In[7]:


selected_company_name


# In[8]:


current_date = datetime.now().strftime('%Y-%m-%d')


# In[9]:


price_data=yf.download(ticker,start='2013-01-01', end=current_date)


# In[10]:


price_data


# In[11]:


price_data.info()


# In[12]:


price_data.isnull().sum()


# In[13]:


# Drop null values
price_data.dropna(inplace=True)


# In[14]:


price_data.describe()


# ### Data Visualization

# In[15]:


import matplotlib.pyplot as plt
# Plot the training set
price_data["Adj Close"][:'2021'].plot(figsize=(16, 4), legend=True)
# Plot the test set
price_data["Adj Close"]['2022':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2021)', 'Test set (2022 and beyond)'])
plt.title(ticker)
plt.show()


# ### Data Preprocessing

# ### MODEL BUILDING

# In[16]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load your historical stock price data into a DataFrame
# Assuming you have a DataFrame named "stock_data" with columns: Date, Open, High, Low, Close, Adj Close, Volume

# Preprocess the data
def preprocess_data(price_data):
    # Drop irrelevant columns and keep only 'Adj Close'
    price_data = price_data[['Adj Close']]

    # Convert the DataFrame to a numpy array
    data = price_data.values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

# Create sequences and their corresponding target values
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

# Split data into train and test sets
def split_data(X, Y, test_size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test

# Define LSTM model
def create_model(seq_length, input_dim):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(seq_length, input_dim)),
        LSTM(units=50),
        Dense(units=1)
    ])
    return model

#preprocess data
scaled_data, scaler = preprocess_data(price_data)

# Define sequence length
seq_length = 30  # You can adjust this value based on your requirements

# Create sequences and their corresponding target values
X, Y = create_sequences(scaled_data, seq_length)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Define input dimensions
input_dim = X_train.shape[2]

# Create and compile model
model = create_model(seq_length, input_dim)
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

# Evaluate model
loss = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print("Root Mean Squared Error (RMSE):", rmse)


# In[17]:


X_test.shape


# In[18]:


import matplotlib.pyplot as plt

# Plot actual and predicted prices
plt.figure(figsize=(18, 10))
plt.plot(price_data.index[-len(Y_test):], Y_test, label='Actual')
plt.plot(price_data.index[-len(predictions):], predictions, label='Predicted')
plt.title('Actual vs Predicted Adj Close Prices')
plt.xlabel('Date')
plt.ylabel('Adj Close Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# ### SENTIMENT ANALYSIS

# In[19]:


import pandas as pd

# Read the Excel file containing news headlines and dates
excel_file = r'D:\DSML PG\CAPSTONE PROJECT\OTHERS DATA\Econ_Times_Archives Jan_2020ToJan_2024.xlsx'

# Read the Excel file containing listed company names
df_companies = pd.read_excel(excel_file, sheet_name='Listed_Comp_NSE')

# Read the data from 'Sheet3' containing news headlines and dates
df_news = pd.read_excel(excel_file, sheet_name='Sheet3')


# In[20]:


df_companies


# In[21]:


df_news


# In[22]:


selected_company_name


# In[85]:


get_ipython().run_cell_magic('time', '', "from transformers import pipeline\n\n# Initialize the sentiment analysis pipeline\nsentiment_analysis_pipeline = pipeline('sentiment-analysis', model='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')\n\n# Function to apply sentiment analysis to a list of headlines\ndef apply_sentiment_analysis(news_headlines):\n    # Initialize an empty list to store sentiment scores and labels\n    sentiment_scores = []\n    sentiment_labels = []\n    # Iterate through each news headline\n    for headline in news_headlines:\n        # Apply sentiment analysis to the headline\n        sentiment_analysis_result = sentiment_analysis_pipeline(headline)[0]\n        # Extract sentiment score and label\n        sentiment_score = sentiment_analysis_result['score']\n        sentiment_label = sentiment_analysis_result['label']\n        # Append the score and label to the respective lists\n        sentiment_scores.append(sentiment_score)\n        sentiment_labels.append(sentiment_label)\n    return sentiment_scores, sentiment_labels\n\n# Function to fetch news headlines for a selected company\ndef fetch_company_news(company_name):\n    # Split the company name into words\n    company_words = company_name.split()\n    # Filter news headlines where at least two words from the company name match\n    return df_news[df_news['News'].apply(lambda headline: sum(word.lower() in headline.lower() for word in company_words) >= 2)]\n\n# Fetch news headlines for the selected company\nrelevant_headlines = fetch_company_news(selected_company_name)\n\n# Apply sentiment analysis to the headlines\nsentiment_scores, sentiment_labels = apply_sentiment_analysis(relevant_headlines['News'])\nrelevant_headlines['Sentiment_Score'] = sentiment_scores\nrelevant_headlines['Sentiment_Label'] = sentiment_labels\n\n# Create a placeholder 'Date' column if not present\nif 'Date' not in relevant_headlines.columns:\n    relevant_headlines['Date'] = pd.to_datetime('today').date()\n\n# Set the date column as the index\nrelevant_headlines.set_index('Date', inplace=True)\n\n# Display the relevant headlines with sentiment scores and labels\nrelevant_headlines\n")


# In[86]:


# Convert the index of the `relevant_headlines` DataFrame to datetime
relevant_headlines.index = pd.to_datetime(relevant_headlines.index)


# In[87]:


relevant_headlines


# In[88]:


price_data


# In[90]:


#COMPARING WITH CHANGE IN STOCK PRICE FOR SAME DAY
# Merge the sentiment scores with the price data based on the date index
merged_data = price_data.merge(relevant_headlines[['Sentiment_Score','Sentiment_Label']], how='left', left_index=True, right_index=True)

# Calculate the stock price change
merged_data['Stock_Price_Change'] = (merged_data['Adj Close'] - merged_data['Open'])

# Calculate the price change percentage
merged_data['Price_Change_Percentage'] = ((merged_data['Adj Close'] - merged_data['Open']) / merged_data['Open']) * 100

# Display the merged data with sentiment classification
merged_data


# In[91]:


merged_data.info()


# In[92]:


# Drop null values
merged_data.dropna(inplace=True)


# In[93]:


merged_data


# ### Compare sentiment score with change in stock price

# In[103]:


from scipy.stats import f_oneway

# Group the data by Sentiment_Label and extract the price change percentage
sentiment_groups = merged_data.groupby('Sentiment_Label')['Price_Change_Percentage']

# Perform ANOVA test
anova_result = f_oneway(*[group for label, group in sentiment_groups])

# Define significance level (alpha)
alpha = 0.05

# Display the ANOVA result
print("ANOVA F-Statistic:", anova_result.statistic)
print("ANOVA p-value:", anova_result.pvalue)

# Check if the p-value is less than or equal to the significance level
if anova_result.pvalue <= alpha:
    print("Reject the null hypothesis.")
    print("There is a statistically significant correlation between sentiment scores and stock price change percentage.")
else:
    print("Do not reject the null hypothesis.")
    print("There is not enough evidence to support a statistically significant correlation between sentiment scores and stock price change percentage.")


# In[107]:


import matplotlib.pyplot as plt

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Sentiment_Label'], merged_data['Price_Change_Percentage'], color='blue', alpha=0.5)
plt.title('Correlation between Sentiment and Price Change Percentage')
plt.xlabel('Sentiment Score')
plt.ylabel('Price Change Percentage')
plt.grid(True)
plt.show()


# In[111]:


merged_data


# ### CORRELATION WITH GOLD PRICE

# In[34]:


# Use the correct ticker symbol for gold, for example, 'GC=F' (Gold Futures)
gold_ticker = 'GC=F'

# Fetch historical data
gold_data = yf.download(gold_ticker, start="2020-01-01", end=current_date)


# In[35]:


gold_data


# In[36]:


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(gold_data['Adj Close'], label='Gold Price', color='gold')
plt.title('Gold Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[37]:


# Rename the 'Adj Close' column in gold_data to avoid column name conflicts after merging
gold_data.rename(columns={'Adj Close': 'Gold_Adj_Close'}, inplace=True)

# Merge gold data into the existing DataFrame based on the date index
merged_data = merged_data.merge(gold_data[['Gold_Adj_Close']], how='left', left_index=True, right_index=True)

# Display the merged DataFrame
merged_data


# In[38]:


merged_data.isnull().sum()


# In[39]:


# Drop null values
merged_data.dropna(inplace=True)


# In[40]:


merged_data.isnull().sum()


# In[41]:


import statsmodels.api as sm


# Fit a linear regression model
X = merged_data[['Gold_Adj_Close']]
y = merged_data['Stock_Price_Change']
X = sm.add_constant(X)  # Add a constant term to the independent variable
model = sm.OLS(y, X).fit()

# Perform hypothesis testing on the coefficient of Gold_Price
p_value = model.pvalues['Gold_Adj_Close']

# Set significance level (alpha)
alpha = 0.05

# Print the results
print("Regression Coefficients:")
print(model.params)
print("\nP-value for Gold_Price Coefficient:", p_value)

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is a statistically significant relationship between gold prices and stock price changes.")
else:
    print("Do not reject the null hypothesis.")
    print("There is not enough evidence to conclude a statistically significant relationship.")


# ### CORRELATION WITH INR VS USD VALUE

# In[42]:


USDINR_ticker = 'USDINR=X'

# Fetch historical data
usd_inr_data = yf.download(USDINR_ticker, start="2020-01-01", end=current_date)


# In[43]:


usd_inr_data


# In[44]:


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(usd_inr_data['Adj Close'], label='usd_inr_data', color='blue')
plt.title('INR VS USD Trend Over Time')
plt.xlabel('Date')
plt.ylabel('INR VALUE(INR)')
plt.legend()
plt.grid(True)
plt.show()


# In[45]:


# Rename the 'Adj Close' column in usd_inr_data to avoid column name conflicts after merging
usd_inr_data.rename(columns={'Adj Close': 'usd_inr_Adj_Close'}, inplace=True)

# Merge gold data into the existing DataFrame based on the date index
merged_data = merged_data.merge(usd_inr_data[['usd_inr_Adj_Close']], how='left', left_index=True, right_index=True)

# Display the merged DataFrame
merged_data


# In[46]:


merged_data.isnull().sum()


# In[47]:


# Drop null values
merged_data.dropna(inplace=True)


# In[48]:


import statsmodels.api as sm


# Fit a linear regression model
X = merged_data[['usd_inr_Adj_Close']]
y = merged_data['Stock_Price_Change']
X = sm.add_constant(X)  # Add a constant term to the independent variable
model = sm.OLS(y, X).fit()

# Perform hypothesis testing on the coefficient of usd_inr value
p_value = model.pvalues['usd_inr_Adj_Close']

# Set significance level (alpha)
alpha = 0.05

# Print the results
print("Regression Coefficients:")
print(model.params)
print("\nP-value for usd_inr_value Coefficient:", p_value)

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is a statistically significant relationship between INR VS USD value and stock price changes.")
else:
    print("Do not reject the null hypothesis.")
    print("There is not enough evidence to conclude a statistically significant relationship.")


# ### CORRELATION WITH FUEL PRICES

# In[49]:


Fuelprice_ticker = 'CL=F'

# Fetch historical data
fuel_price_data = yf.download(Fuelprice_ticker, start="2020-01-01", end=current_date)


# In[50]:


fuel_price_data


# In[51]:


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(fuel_price_data['Adj Close'], label='fuel_price_data', color='grey')
plt.title('Fuel Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('FUEL PRICE(USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[52]:


# Rename the 'Adj Close' column in fuel_price_data to avoid column name conflicts after merging
fuel_price_data.rename(columns={'Adj Close': 'fuel_price_Adj_Close'}, inplace=True)

# Merge gold data into the existing DataFrame based on the date index
merged_data = merged_data.merge(fuel_price_data[['fuel_price_Adj_Close']], how='left', left_index=True, right_index=True)

# Display the merged DataFrame
merged_data


# In[53]:


merged_data.isnull().sum()


# In[54]:


# Drop null values
merged_data.dropna(inplace=True)


# In[55]:


# Fit a linear regression model
X = merged_data[['fuel_price_Adj_Close']]
y = merged_data['Stock_Price_Change']
X = sm.add_constant(X)  # Add a constant term to the independent variable
model = sm.OLS(y, X).fit()

# Perform hypothesis testing on the coefficient of Fuel price
p_value = model.pvalues['fuel_price_Adj_Close']

# Set significance level (alpha)
alpha = 0.05

# Print the results
print("Regression Coefficients:")
print(model.params)
print("\nP-value for Fuel_Price Coefficient:", p_value)

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is a statistically significant relationship between Fuel price and stock price changes.")
else:
    print("Do not reject the null hypothesis.")
    print("There is not enough evidence to conclude a statistically significant relationship.")


# In[56]:


merged_data.head()


# In[ ]:





# In[ ]:





# In[ ]:




