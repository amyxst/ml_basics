import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style # make the plot look decent? 

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # will treat values as outliers, don't want to sacrifice data unnecessarily

forecast_out = int(math.ceil(0.01*len(df))) # 10% of the data points

df['Label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)

print(df.head())

X = np.array(df.drop(['Label'], 1))
# X = preprocessing.scale(X)
# y = np.array(df['label'])

X = preprocessing.scale(X) # need to scale new values WITH training data for classifier to work properly
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train) # synonymous w fit
accuracy = clf.score(X_test, y_test) # synonymous with score

# print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name # iloc is indexing for dataframes (?)
print('last date {}'.format(last_date))
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

print("first {}".format(df.tail()))
print(df.head())
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] # 
	# print(df.loc[next_date])

print("second {}".format(df.tail()))

# df['Adj. Close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show() 