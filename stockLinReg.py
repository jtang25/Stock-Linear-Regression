import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mplcyberpunk

df= pd.read_csv(r"C:\Users\Jason_yj4bjfp\IdeaProjects\Stock-Linear-Regression\SOXL.csv")

def linregstock(df, start_date, end_date):
    plt.figure(figsize=(20, 7), dpi=80)
    plt.style.use('cyberpunk')

    plt.plot(df['Date'],df['Close'])
    font2 = {'size':20}
    plt.xlabel('Date', fontdict = font2)
    plt.ylabel('Price', fontdict = font2)

    r = df['Date'].tolist()
    df = df.drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
    datelist = pd.date_range(start=start_date,end=end_date).tolist()
    a = pd.DataFrame(datelist)[0].tolist()
    del  a[0]
    r = r + a
    r=pd.Series(r)
    e = pd.DataFrame(pd.to_datetime(r).rsub(pd.Timestamp(start_date).floor('d')).dt.days.abs())

    df['new'] = pd.to_datetime(r).rsub(pd.Timestamp(start_date).floor('d')).dt.days.abs()
    Y = df.drop(['new','Date'], axis = 1)
    X = df[['new']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    y_predict = regression_model.predict(e)

    plt.plot(e, y_predict, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')
    plt.show()
    return regression_model

print(linregstock(df, "2022-11-29", "2023-02-17"))
