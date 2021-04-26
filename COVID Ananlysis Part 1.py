#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[4]:


covid_data = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')


# In[8]:


covid_data.head()


# In[9]:


covid_data.describe()


# In[10]:




state_wise = covid_data.groupby('State/UnionTerritory')['Confirmed','Cured','Deaths'].sum().reset_index()
state_wise["Death_percentage"] = ((state_wise["Deaths"] / state_wise["Confirmed"]) * 100)
state_wise.style.background_gradient(cmap='magma')


# In[11]:


px.bar(x=state_wise.nlargest(10,"Confirmed")["State/UnionTerritory"],
       y = state_wise.nlargest(10,"Confirmed")["Confirmed"],
       color_discrete_sequence=px.colors.diverging.Picnic,
       title="Top 10 states with highest number of Confirmed cases")


# In[12]:


px.bar(x=state_wise.nlargest(10,"Cured")["State/UnionTerritory"],
       y = state_wise.nlargest(10,"Cured")["Cured"],
       color_discrete_sequence=px.colors.sequential.Sunset,
       title="Top 10 states with highest number of Cured cases")


# In[13]:


px.bar(x=state_wise.nlargest(10,"Deaths")["State/UnionTerritory"],
       y = state_wise.nlargest(10,"Deaths")["Deaths"],
       color_discrete_sequence=px.colors.diverging.curl,
       title="Top 10 states with highest number of Deaths")


# In[14]:


px.bar(x=state_wise.nlargest(10,"Death_percentage")["State/UnionTerritory"],
       y = state_wise.nlargest(10,"Death_percentage")["Death_percentage"],
       color_discrete_sequence=px.colors.diverging.Portland,
       title="Top 10 states with highest of Death percentage")


# In[50]:


month_wise = covid_19_indiadata.groupby(pd.Grouper(key='Date',freq='M')).sum()

month_wise = month_wise.drop(['Sno'], axis = 1)
month_wise['Date'] = month_wise.index

first_column = month_wise.pop('Date')
month_wise.insert(0, 'Date', first_column)

index = [x for x in range(len(month_wise))]
month_wise['index'] = index
month_wise = month_wise.set_index('index')

second_column = month_wise.pop('Confirmed')
month_wise.insert(1, 'Confirmed', second_column)
month_wise["Death_percentage"] = ((month_wise["Deaths"] / month_wise["Confirmed"]) * 100)
month_wise.style.background_gradient(cmap='twilight_shifted')


# In[2]:


covid_data = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[4]:


covid_data = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')


# In[6]:


covid_testing = pd.read_csv(r'C:\Users\shubham.kj\Desktop\StatewiseTestingDetails.csv')
covid_testing['Date'] = covid_testing['Date'].astype('datetime64[ns]')
covid_testing.head()


# In[7]:


covid_testing_state = covid_testing.groupby('State')['TotalSamples','Negative','Positive'].max().reset_index()
covid_testing_state["Positive_percentage"] = ((covid_testing["Positive"] / covid_testing["TotalSamples"]) * 100)
covid_testing_state.style.background_gradient(cmap='gist_earth_r')//Statewise Testing Analysis


# In[8]:


px.bar(x=covid_testing_state.nlargest(10,"TotalSamples")["State"],
       y = covid_testing_state.nlargest(10,"TotalSamples")["TotalSamples"],
       labels={'y':'Total Samples','x':'State'},
       color_discrete_sequence=px.colors.sequential.haline,
       title="Top 10 states with highest number of Total Samples")


# In[11]:


px.bar(x=covid_testing_state.nlargest(10,"Positive")["State"],
       y = covid_testing_state.nlargest(10,"Positive")["Positive"],
       labels={'y':'Total Positive Cases','x':'State'},
       color_discrete_sequence=px.colors.sequential.solar,
       title="Top 10 states with highest number of Positive cases")


# In[12]:


px.bar(x=covid_testing_state.nlargest(10,"Positive_percentage")["State"],
       y = covid_testing_state.nlargest(10,"Positive_percentage")["Positive_percentage"],
       labels={'y':'Positive Percentage','x':'State'},
       color_discrete_sequence=px.colors.sequential.Aggrnyl,
       title="Top 10 states with highest Positive percentage",
       height = 420)


# In[15]:


deaths = covid_data.groupby("State/UnionTerritory")["Deaths"].sum().reset_index()


# In[16]:


deaths.head()


# In[17]:


px.treemap(deaths,path=["State/UnionTerritory"],values="Deaths",title="Overall States Comparision of deaths")


# In[20]:


covid_vaccine_statewise = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_vaccine_statewise.csv')


# In[21]:


covid_vaccine_statewise.head()


# In[23]:


mask = (covid_vaccine_statewise["State"]=="India")
x3 = covid_vaccine_statewise[mask]["Updated On"]


# In[24]:


y3 = covid_vaccine_statewise[mask]["Total Individuals Registered"]


# In[25]:


fig = px.line(x = x3, y = y3,color_discrete_sequence=px.colors.qualitative.Dark2,
       title="Total Individuals Registering for vaccines")

fig.update_xaxes(title_text="Dates")
fig.update_yaxes(title_text="Number of Persons Registered")

fig.show()


# In[26]:


y5 = covid_vaccine_statewise[mask]["First Dose Administered"]
y6 = covid_vaccine_statewise[mask]["Second Dose Administered"]


# In[27]:


fig = px.line(x = x3, y = [y5,y6],color_discrete_sequence=px.colors.qualitative.Dark2,
       title="First Dose and Second Dose Administered Comparison all over India")

fig.update_xaxes(title_text="Dates")
fig.update_yaxes(title_text="No of persons given vaccines")


# In[28]:


y7 = covid_vaccine_statewise[mask]["Male(Individuals Vaccinated)"]
y8 = covid_vaccine_statewise[mask]["Female(Individuals Vaccinated)"]
y9 = covid_vaccine_statewise[mask]["Transgender(Individuals Vaccinated)"]


# In[29]:


px.line(x = x3, y = [y7,y8,y9],color_discrete_sequence=px.colors.qualitative.Dark2,
       title="Male vs Female vaccinated in India")


# In[30]:


y10 = covid_vaccine_statewise[mask]["Total Covaxin Administered"]
y11 = covid_vaccine_statewise[mask]["Total CoviShield Administered"]


# In[31]:


px.line(x = x3, y = [y10,y11],color_discrete_sequence=px.colors.qualitative.Dark2,
       title="Male vs Female vaccinated in India")


# In[41]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px
from collections import namedtuple


# In[42]:


pip install fbprophet


# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[46]:


covid_data['Date'] = covid_data['Date'].astype('datetime64[ns]')
covid_data.head()


# In[51]:


month_wise = covid_data.groupby(pd.Grouper(key='Date',freq='M')).sum()

month_wise = month_wise.drop(['Sno'], axis = 1)
month_wise['Date'] = month_wise.index

first_column = month_wise.pop('Date')
month_wise.insert(0, 'Date', first_column)

index = [x for x in range(len(month_wise))]
month_wise['index'] = index
month_wise = month_wise.set_index('index')

second_column = month_wise.pop('Confirmed')
month_wise.insert(1, 'Confirmed', second_column)
month_wise["Death_percentage"] = ((month_wise["Deaths"] / month_wise["Confirmed"]) * 100)
month_wise.style.background_gradient(cmap='twilight_shifted')//Month Wise Shift


# In[53]:


fig = px.bar(month_wise, x='Date', y='Confirmed',
             hover_data=['Cured', 'Deaths'], color='Date',
             labels={'Date':'Date(monthwise)'},
             title="Monthwise Increase in Confirmed cases")
fig.show()


# In[54]:


fig = px.bar(month_wise, x='Date', y='Cured',
             hover_data=['Confirmed','Deaths'], color='Date',
             labels={'Date':'Date(monthwise)'},
             title="Monthwise Increase in Cured cases")
fig.show()


# In[55]:


fig = px.bar(month_wise, x='Date', y='Deaths',
             hover_data=['Confirmed','Cured'], color='Date',
             labels={'Date':'Date(monthwise)'},
             title="Monthwise Increase in Deaths cases")
fig.show()


# In[58]:


fig = px.bar(month_wise , 
             x='Date', 
             y='Death_percentage' ,
             hover_data=['Confirmed','Deaths'],color='Date',
             labels={'Death_percentage':'Death percentage'},
             title="Top 10 states with highest of Death percentage")
fig.show()


# In[60]:


covid_data.hist(figsize=(20,15),edgecolor='black');


# In[62]:


plt.figure(figsize=(20,10))
sns.lineplot(data=covid_data, y="Deaths",x='Cured')
plt.title('Relation betwwen Deaths and Cured');


# In[64]:


pip install plot


# In[65]:


import plot


# In[66]:


plot(covid_data,'cured')


# In[67]:


#default theme
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[15,10]
matplotlib.rcParams.update({'font.size': 15})


# In[68]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[70]:


plt.figure(figsize=(20,10))
sns.lineplot(data=covid_data, y="Deaths",x='Cured')
plt.title('Relation betwwen Deaths and Cured');


# In[86]:


covid_data2 = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_data2.csv')


# In[88]:


covid_data2.head()


# In[90]:


cases = pd.read_csv(r'C:\Users\shubham.kj\Desktop\cases.csv')


# In[91]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[92]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix


# In[93]:


plt.style.use('fivethirtyeight')


# In[95]:


df = pd.read_csv(r'C:\Users\shubham.kj\Desktop\cases.csv')


# In[96]:


df.head()


# In[97]:


df.info()


# In[98]:


df[df.duplicated()]


# In[99]:


df.drop_duplicates(inplace=True)


# In[100]:


df.info()


# In[101]:


df.describe()


# In[107]:


#check correlation to idenitify what has more correlation with the out

plt.figure(figsize=(32,16))
sns.heatmap(covid_data2.corr(),annot=True)
plt.show()


# In[108]:


#split the data to feature and label

X = covid_data2.iloc[:, 0: -1]
y = covid_data2.iloc[:, -1:]


# In[114]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)


# In[115]:


import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import SGD
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential


# In[ ]:


pip install tensorflow


# In[ ]:




