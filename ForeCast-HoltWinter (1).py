#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


features = pd.read_csv('Features data set.csv')
sales = pd.read_csv('sales data-set.csv') 
stores = pd.read_csv('stores data-set.csv')


# In[4]:


features.sample(n=5)


# In[5]:


sales.sample(n=5)


# In[6]:


stores.sample(n=5)


# In[7]:


features = features.merge(stores, on = 'Store')
df = features.merge(sales, on = ['Store','Date','IsHoliday'])
df=df.fillna(0)


# In[8]:


df.shape


# In[9]:


df.sample(5)


# In[10]:


df.describe()


# In[11]:


df = df.sort_values(by='Date')


# In[12]:


df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


# In[13]:


df = df.set_index('Date')


# In[14]:


df['Weekly_Sales'].plot(figsize=(25,8));


# In[15]:


df_Sales = df[['Weekly_Sales']]


# In[16]:


df_Sales.head()


# In[17]:


df_Sales = df_Sales.resample(rule='M').mean()


# In[18]:


df_Sales.head()


# In[19]:


df_Sales = df_Sales.rename(columns={'Weekly_Sales':'Monthly_Sales'})


# In[20]:


df_Sales.plot(figsize=(20,8))
plt.title('Average Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Dollar Sales');


# In[21]:


df_Sales.isnull().sum()


# In[22]:


df.to_csv('Retail Sales Monthly.csv',index=False)


# In[23]:


df_Sales.shape


# In[24]:


sales_train = df_Sales.iloc[:22]
sales_test = df_Sales.iloc[21:]


# In[25]:


sales_test


# In[26]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# fitted_model = ExponentialSmoothing(sales_train['Monthly_Sales'],
#                                    trend = 'add',
#                                    seasonal = 'add',
#                                    seasonal_periods = 10).fit()

# In[30]:


fitted_model = ExponentialSmoothing(sales_train['Monthly_Sales'],
                                   trend = 'add',
                                   seasonal = 'add',
                                   seasonal_periods = 10).fit()


# In[31]:


test_predictions = fitted_model.forecast(24)


# In[32]:


sales_train['Monthly_Sales'].plot(legend=True, label= 'TRAIN', figsize=(15,8))
sales_test['Monthly_Sales'].plot(legend=True, label= 'TEST', figsize=(15,8))
test_predictions.plot(legend=True, label= 'PREDICTIONS', figsize=(15,8))

plt.title('Train, Test, and Sales Predictions')
plt.xlabel("Date")
plt.ylabel("Dollar Sales");


# In[33]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[34]:


sales_test.describe()


# In[35]:


test_predictions = fitted_model.forecast(15)


# In[36]:


MSE = mean_squared_error(sales_test, test_predictions)

MAE = mean_absolute_error(sales_test, test_predictions)

RMSE = np.sqrt(mean_squared_error(sales_test, test_predictions))

pd.options.display.float_format = '{:.2f}'.format

results = pd.DataFrame({'Squared Error': ['MSE','MAE','RMSE','STD DVTN'],
                       'Score': [MSE,MAE,RMSE, '1047']})
results = results.set_index('Squared Error')
results


# In[37]:


import pickle


# In[40]:


filename='ForeCast-HoltWinter.sav'
pickle.dump(fitted_model, open(filename,'wb'))


# In[41]:


loadmodel = pickle.load(open('ForeCast-HoltWinter.sav', 'rb'))


# In[42]:


test_predictions = fitted_model.forecast(24)


# In[43]:


sales_train['Monthly_Sales'].plot(legend=True, label= 'TRAIN', figsize=(15,8))
sales_test['Monthly_Sales'].plot(legend=True, label= 'TEST', figsize=(15,8))
test_predictions.plot(legend=True, label= 'PREDICTIONS', figsize=(15,8))

plt.title('Train, Test, and Sales Predictions')
plt.xlabel("Date")
plt.ylabel("Dollar Sales");


# In[ ]:




