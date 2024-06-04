# Udacity BostonAirBnB
## Udacity project: Writing a Data Scientist Blog Post

### This project is related to Boston AirBnB dataset and deliverables of the project is to come up with at least 3 business questions to be answered by data analysis.

### These are the imports I used for the analysis:

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
### The next step is to read files and check the info.

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

```
### Compared to the rest of columns, price has around half values missing. This is the case since if a listing is not available, there's no price. For purposes of this analysis, I'm interested only in listings that are available, so I'm going to drop rows with missing price values. Also, date needs to be changed from object type to datetime and price needs to be an integer, since all the values 

```
calendar_c = calendar.dropna()

