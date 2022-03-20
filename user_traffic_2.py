import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import streamlit as st

st.set_page_config(layout="wide")

data = pd.read_excel('Daily Users.xlsx').copy()
data['Date'] = pd.to_datetime(data['Date'], format="%b %d '%y")
data['Day'] = data['Date'].apply(lambda x: x.strftime('%A'))
data['Month'] = data['Date'].apply(lambda x: x.strftime('%B'))

st.title('User Traffic Forecasting')

# -----------------------------------------------------------------------------------
# Graphs

def plot_data(data, x, y, title, xlabel, ylabel, dpi, two_sided=False):
    fig = plt.figure(figsize=(16,5), dpi=dpi)
    if two_sided:
        plt.fill_between(x=x, y1=y, y2=-y, alpha=0.5, linewidth=2,color='seagreen')
        plt.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=0.5)
        title = title + '(Two Sided)'
    else:
        plt.plot(x, y)  
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    return st.pyplot(fig)

st.header('Graphs')
st.subheader('User Traffic')
plot_data(data=data, x=data['Date'], y=data['User Traffic'], title='User Traffic', xlabel='Date', ylabel='Traffic', dpi=100)

st.subheader('User Traffic (Two Sided)')
plot_data(data=data, x=data['Date'], y=data['User Traffic'], title='User Traffic', xlabel='Date', ylabel='Traffic', dpi=150, two_sided=True)

st.subheader('Daily User Traffic')
months = data['Month'].unique()
fig = plt.figure(figsize=(16,5), dpi=100)
for month in months:
    traffic = data.loc[data['Month']==month, 'User Traffic']
    date = [i for i in range(1,len(traffic)+1)]
    plt.plot(date, traffic, label=month)
plt.legend()
plt.xlabel('Date of the month')
plt.ylabel('User Traffic')
plt.title('Daily User Traffic')
st.pyplot(fig)

st.subheader('Day-wise Box Plot')
fig = plt.figure(figsize=(8,3), dpi=100)
sns.boxplot(x='Day', y='User Traffic', data=data)
plt.xticks(rotation=90)
plt.title('Day-wise Box Plot')
st.pyplot(fig)

st.subheader('Day-wise User Traffic')
days = data['Day'].unique()
daywise_traffic = []
for day in days:
    traffic = sum(data.loc[data['Day']==day, 'User Traffic'])
    daywise_traffic.append(traffic)
fig = plt.figure(figsize=(8,3), dpi=100)
plt.bar(x=days, height=daywise_traffic, color='orange')
plt.title('Day-wise User Traffic')
st.pyplot(fig)

# -----------------------------------------------------------------------------------
# Time series forecasting
st.header('User Traffic Forecasting')

n_steps = st.slider('Number of steps for forecasting', 1, 14, 8)

# ARMA Model
model = ARIMA(data['User Traffic'], order=(n_steps,0,n_steps)) # (AR order, I order, MA order)
model = model.fit()

y = data['User Traffic'][n_steps:]
y_pred = model.predict(start=n_steps,end=len(data)-1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

def predict(date, data, n_steps):
    date = datetime.strptime(date, '%Y-%m-%d')
    last_date = str(data['Date'].values[-1]).split('T')[0]
    last_date = datetime.strptime(last_date, '%Y-%m-%d')
    min_date = str(data['Date'].values[0]).split('T')[0]
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    
    if min_date<=date<=last_date:
        traffic = data[data['Date']==date]
        traffic = traffic['User Traffic']
        traffic = list(traffic)[0]
    elif date>last_date:
        delta = date-last_date
        delta = delta.days
        traffic = model.predict(start=len(data)-1,end=len(data)-1+delta)
        traffic = list(traffic)[-1]
    else:
        return 'Error'
    
    return traffic

date = st.date_input('Date')
date = date.strftime('%Y-%m-%d')
if st.button('Predict User Traffic'):
    output = predict(date, data, n_steps=n_steps)
    st.write(f"""
    User Traffic on {date} is **{int(output-rmse)}** and **{int(output+rmse)}** users.
    """)
