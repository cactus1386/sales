import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv('Salary_dataset.csv')

df.head()

cdf = df[['YearsExperience', 'Salary']]
cdf.head(10)

chrt = cdf[['YearsExperience', 'Salary']]
chrt.hist()
plt.show()

plt.scatter(cdf.YearsExperience, cdf.Salary, color="blue")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.YearsExperience, train.Salary, color='blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

from sklearn import linear_model
reg = linear_model.LinearRegression()
trainx = np.asanyarray(train[['YearsExperience']])
trainy = np.asanyarray(train[['Salary']])
reg.fit (trainx, trainy)

print(reg.coef_)
print(reg.intercept_)

plt.scatter(train.YearsExperience, train.Salary, color='blue')
plt.plot(trainx, reg.coef_[0][0]*trainx + reg.intercept_[0], '-r')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import r2_score

testx = np.asanyarray(test[['YearsExperience']])
testy = np.asanyarray(test[['Salary']])
testy_ = reg.predict(testx)

print("Mean absolute error: %.2f" % np.mean(np.absolute(testy_ - testy)))
print("Residual sum of squares (MSE): %.2f" % np.mean((testy_ - testy) ** 2))
print("R2-score: %.2f" % r2_score(testy , testy_) )