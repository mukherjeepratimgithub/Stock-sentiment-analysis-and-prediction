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


# In[51]:


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


# In[34]:


ticker


# In[35]:


selected_company_name


# In[6]:


current_date = datetime.now().strftime('%Y-%m-%d')


# In[7]:


price_data=yf.download(ticker,start='2013-01-01', end=current_date)


# In[8]:


price_data


# In[9]:


price_data.info()


# In[10]:


price_data.isnull().sum()


# In[11]:


# Drop null values
price_data.dropna(inplace=True)


# In[12]:


price_data.describe()


# ### Data Visualization

# In[13]:


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

# In[14]:


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


# In[15]:


X_test.shape


# In[16]:


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

# In[52]:


import pandas as pd

# Read the Excel file containing news headlines and dates
excel_file = r'D:\DSML PG\CAPSTONE PROJECT\OTHERS DATA\Econ_Times_Archives Jan_2020ToJan_2024.xlsx'

# Read the Excel file containing listed company names
df_companies = pd.read_excel(excel_file, sheet_name='Listed_Comp_NSE')

# Read the data from 'Sheet3' containing news headlines and dates
df_news = pd.read_excel(excel_file, sheet_name='Sheet3')


# In[53]:


df_companies


# In[54]:


df_news


# In[55]:


selected_company_name


# In[56]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

relevant_headlines = []
# Function to fetch news headlines for a selected company
def fetch_company_news(company_name):
    # Split the company name into words
    company_words = company_name.split()
    # Filter news headlines where at least two words from the company name match
    return df_news[df_news['News'].apply(lambda headline: sum(word.lower() in headline.lower() for word in company_words) >= 2)]

# Fetch news headlines for the selected company
relevant_headlines = fetch_company_news(selected_company_name)

# Create a placeholder 'Date' column if not present
if 'Date' not in relevant_headlines.columns:
    relevant_headlines['Date'] = pd.to_datetime('today').date()
# Calculate sentiment score for each headline
relevant_headlines['Sentiment'] = relevant_headlines['News'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Set the date column as the index
relevant_headlines.set_index('Date', inplace=True)

# Display the relevant headlines with sentiment scores
print(relevant_headlines)


# In[57]:


relevant_headlines


# In[58]:


# Convert the index of the `relevant_headlines` DataFrame to datetime
relevant_headlines.index = pd.to_datetime(relevant_headlines.index)


# In[59]:


relevant_headlines


# In[60]:


price_data


# In[61]:


# Merge the sentiment scores with the price data based on the date index
merged_data = price_data.merge(relevant_headlines['Sentiment'], how='left', left_index=True, right_index=True)
# Define threshold values for sentiment classification
positive_threshold = 0.05
negative_threshold = -0.05

# Classify sentiment scores into positive, neutral, and negative categories
def classify_sentiment(score):
    if score > positive_threshold:
        return 'Positive'
    elif score < negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment classification to the 'Sentiment' column
merged_data['Sentiment_Class'] = merged_data['Sentiment'].apply(classify_sentiment)
# Calculate the price change percentage
merged_data['Price_Change_Percentage'] = ((merged_data['Adj Close'] - merged_data['Open']) / merged_data['Open']) * 100

# Display the merged data with sentiment classification
merged_data


# In[62]:


# Drop null values
merged_data.dropna(inplace=True)


# In[63]:


merged_data


# ### Compare sentiment score with change in stock price

# In[64]:


from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(merged_data['Sentiment'], merged_data['Price_Change_Percentage'])

# Define significance level (alpha)
alpha = 0.05

# Display the results
print("Pearson correlation coefficient:", correlation_coefficient)
print("p-value:", p_value)

# Check if the p-value is less than or equal to the significance level
if p_value <= alpha:
    print("Reject the null hypothesis.")
    print("There is a statistically significant correlation between sentiment scores and stock price change percentage.")
else:
    print("Do not reject the null hypothesis.")
    print("There is not enough evidence to support a statistically significant correlation between sentiment scores and stock price change percentage.")


# In[65]:


import matplotlib.pyplot as plt

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Sentiment'], merged_data['Price_Change_Percentage'], color='blue', alpha=0.5)
plt.title('Correlation between Sentiment and Price Change Percentage')
plt.xlabel('Sentiment Score')
plt.ylabel('Price Change Percentage')
plt.grid(True)
plt.show()


# In[66]:


# Calculate correlation between sentiment score and stock price change for all days
correlation_all_days = merged_data['Sentiment'].corr(merged_data['Adj Close'].pct_change())

# Display the correlation for all days
print("Correlation between sentiment score and stock price change for all days:", correlation_all_days)


# In[ ]:





# In[ ]:




