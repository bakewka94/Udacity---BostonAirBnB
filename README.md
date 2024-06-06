# Udacity BostonAirBnB
## Udacity project: Writing a Data Scientist Blog Post

### This project is related to Boston AirBnB dataset and deliverables of the project is to come up with at least 3 business questions to be answered by data analysis. I've approached this dataset with questions I would have if I were planning a trip to Boston and was considering using AirBnB instead of a hotel. 

Questions I'm trying to answer by this analysis:

1.	How are prices affected by seasonality? Would the summer months be the most expensive ones as they tend to? What about days of the week – are weekends the most expensive period?
2.	What are the best properties to consider booking? 
3.	By looking at variables provided in the listing dataset, I believe number of rooms, bathrooms, property type and room type would be a best set of variables to predict the price of a listing. Would this be the case in a linear regression model?



These are the imports I used for the analysis:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
```
The next step is to read files and check the info.

```
listings = pd.read_csv("OneDrive - Chevron/Desktop/Data Science/Udacity/Boston Airbnb/listings.csv")
calendar = pd.read_csv("OneDrive - Chevron/Desktop/Data Science/Udacity/Boston Airbnb/calendar.csv")
reviews = pd.read_csv("OneDrive - Chevron/Desktop/Data Science/Udacity/Boston Airbnb/reviews.csv")
```

```
calendar.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1308890 entries, 0 to 1308889
Data columns (total 4 columns):
 #   Column      Non-Null Count    Dtype 
---  ------      --------------    ----- 
 0   listing_id  1308890 non-null  int64 
 1   date        1308890 non-null  object
 2   available   1308890 non-null  object
 3   price       643037 non-null   object
dtypes: int64(1), object(3)
memory usage: 39.9+ MB


The calendar dataset has missing prices values for those listings that are not available at the time. I’ve not seen a metadata that would specify it these listings are occupied or just not being rented out by the owner. For purposes of this analysis, I’m dropping these rows. Price column needs to be changed from object to integer, date column to be changed from object to datetime. I also want to extract month and day of the week from dates into separate columns.

```
calendar_c['price'] = calendar_c['price'].replace('[\$,]', '', regex=True).astype(float).astype('int64')
calendar_c['date'] = pd.to_datetime(calendar_c['date'])
calendar_c['month'] = calendar_c['date'].dt.month
```

Now, to answer first question let’s group up the average price of listings per month.

```
avg_price_month = calendar_c.groupby('month')['price'].mean()
print(avg_price_month.sort_values(ascending=False))
month
9     237.047727
10    233.416248
8     203.330142
11    202.924416
7     202.486309
4     197.252890
6     196.535302
5     193.712295
12    192.601915
1     182.799671
3     181.818742
2     180.961028
Name: price, dtype: float64
```

```
avg_price_month.plot(x='month', y = 'price', kind = 'bar')
```

![Distribution of best apartments per neighborhood](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/40ace3f5-97fd-4a55-8ded-f3d7302f1573)








