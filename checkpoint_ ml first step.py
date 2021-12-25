#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("kc_house_data.csv")
data.head(100)


# In[2]:


data.info()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
g = sns.FacetGrid(data, col="grade")
g.map(plt.hist, 'price', bins = 20)
g


# In[4]:


g = sns.FacetGrid(data, col = "bedrooms")
g.map(plt.hist, "price", bins = 20)
g


# In[5]:


g = sns.FacetGrid(data, col = "condition")
g.map(plt.hist, "price", bins = 20)


# In[6]:


g = sns.FacetGrid(data, col = "floors")
g.map(plt.hist, "price", bins = 20)


# In[7]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 22 , 20 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : 1.2 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[8]:


plot_correlation_map(data)


# In[9]:


# strong correlation between price and sqft-living


# In[35]:


from sklearn.model_selection import train_test_split
y = data["price"].values
x = data["sqft_living"].values[:,np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 30, test_size = .2)


# In[36]:


plt.scatter(x,y, color="r")
plt.plot(x, model.predict(x), color="k")
plt.show()


# In[37]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
model = LinearRegression()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("mse", mean_squared_error(y_test, predicted))
print("r²", metrics.r2_score(y_test, predicted))


# In[38]:


x = data[["sqft_living", "grade"]].values
y = data["price"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .35, random_state = 40)
model = LinearRegression()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("mse", mean_squared_error(y_test, predicted))
print("r²", metrics.r2_score(y_test, predicted))


# In[39]:


from sklearn.preprocessing import PolynomialFeatures
x = data[["sqft_living", "grade"]].values
y = data["price"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .35, random_state = 40)
lg = LinearRegression()
poly = PolynomialFeatures(degree=3)


# In[21]:


x_train_fit = poly.fit_transform(x_train)
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)
print("mse", mean_squared_error(y_test, predicted))
print("r²", metrics.r2_score(y_test, predicted))


# In[ ]:


#polynomial regression has smaller mse and greater R², best model

