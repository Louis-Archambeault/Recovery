import numpy as np
import matplotlib.pyplot as plt #Data visualisation libraries
import seaborn as sns
import pandas as pd



RecData = pd.read_csv('Recdata.csv')
RecData.head()
RecData.info()
RecData.describe()
RecData.columns


#sns.pairplot(RecData)
#plt.show()


#sns.distplot(RecData['Recovery'])
#plt.show()

#RecData.corr()

X = RecData[['Mo','Nb','Zr','Sr','Rb','Th','Pb','Ars','Ta','Zn','Cu',
             'Ni','Co','Fe','Mn','Cr','V','Ti','Sc','Ca','K','S','Ba',
             'Cs','Ag','Nb_Zr','V_Zr','Ti_Zr','Sc_Zr','Sr_Zr','Ta_Zr',
             'Ca_Zr','K_Zr','Rb_Zr','Mn_Zr','Cs_Zr','Zn_Zr','Fe_Zr',
             'Ni_Zr','Co_Zr','Cu_Zr',
             'Au_Total','MagSus','Au','Au_Tails']]


y = RecData['Recovery']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()


