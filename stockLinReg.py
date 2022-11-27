import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linregstock(df, start_date, end_date):
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
    return y_predict
