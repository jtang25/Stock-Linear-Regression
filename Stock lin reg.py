import requests
import seaborn as sns
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew
import mplcyberpunk
content_key = [REDACTED]

plt.figure(figsize=(20, 7), dpi=80)
plt.style.use('cyberpunk')

df = pd.read_csv("SOXL (2).csv")
df['Date'] = pd.to_datetime(df['Date'])
df.drop(['Open','High','Low','Adj Close','Volume'], axis=1, inplace=True)
plt.plot(df['Date'],df['Close'])
font2 = {'size':20}
plt.xlabel('Date', fontdict = font2)
plt.ylabel('Price', fontdict = font2)

r = df['Date'].tolist()
datelist = pd.date_range(start="2022-07-01",end="2022-08-26").tolist()
a = pd.DataFrame(datelist)[0].tolist()
del  a[0]
r=r+a
r=pd.Series(r)
e = pd.DataFrame(pd.to_datetime(r).rsub(pd.Timestamp('2022-07-01').floor('d')).dt.days.abs())

df['new'] = pd.to_datetime(r).rsub(pd.Timestamp('2022-07-01').floor('d')).dt.days.abs()
Y = df.drop(['new','Date'], axis = 1)
X = df[['new']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_predict = regression_model.predict(e)

plt.plot(r, y_predict, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')
print(regression_model.coef_[0][0])
mplcyberpunk.make_lines_glow()
mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
