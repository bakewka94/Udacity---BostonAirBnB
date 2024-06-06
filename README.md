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
```

The calendar dataset has missing prices values for those listings that are not available at the time. I’ve not seen a metadata that would specify it these listings are occupied or just not being rented out by the owner. For purposes of this analysis, I’m dropping these rows. Price column needs to be changed from object to integer, date column to be changed from object to datetime. I also want to extract month and day of the week from dates into separate columns.

```
calendar_c['price'] = calendar_c['price'].replace('[\$,]', '', regex=True).astype(float).astype('int64')
calendar_c['date'] = pd.to_datetime(calendar_c['date'])
calendar_c['month'] = calendar_c['date'].dt.month

```

### To answer first question let’s group up the average price of listings per month.

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
![Price per month](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/ab6c83d3-41ae-4a5b-815d-c65665dd2a17)


Turns out my assumption was wrong. Period from August to November is the most expensive one with September being top 1. Interesting to see that November has higher prices than July or June. I'm not a fan of colder months, so if I were to visit I'd choose May-June.

Now let’s check prices for days of the week.

```
calendar_c['day_of_week'] = calendar_c['date'].dt.day_name()
avg_price_day_of_week = calendar_c.groupby('day_of_week')['price'].mean()
print(avg_price_day_of_week.sort_values(ascending = False))
day_of_week
Saturday     203.408387
Friday       203.121167
Sunday       198.219764
Thursday     198.073112
Monday       195.809561
Wednesday    195.418228
Tuesday      195.173842
Name: price, dtype: float64
```

```
ax = avg_price_day_of_week.sort_values(ascending = False).plot(x='day_of_week', y = 'price', kind = 'bar')
ax.set_ylabel ('Average Price')
ax.set_xlabel('Day of the week')
```
![Price per day of week](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/326a1507-041c-46b7-94f1-759f64e13efc)


As expected, weekends are the most expensive. This being said the difference is not that high – only $8 between the highest and lowest prices. If I lived close to Boston area and wanted to visit the city, going there on weekends still might be a good idea as saving $8 per day should be worth the conveience of travelling on the weekend.

### Now, onto the next question - What are the best properties to consider booking? 

If you’re anything like me, you’d go for listings with highest scores and you’d look for the average price within this best category. I’d define it here using the overall rating – those with 90 or higher. Let’s start by inspecting average price per type of property and their count. 

```
best = listings[listings['review_scores_rating'] >= 90]
listings.shape
best.shape
```
We went from 3585 listings to 2052. Similar to the calendar dataset, this also requires some preprocessing. Namely, price column’s datatype needs to be converted. 

```
best['price'] = best['price'].replace('[\$,]', '', regex=True).astype(float).astype('int64')
best.groupby('property_type').agg({'price': ['mean', 'count']}).sort_values(by=('price', 'mean'), ascending=False)
```
![best_group_by](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/637d7fdd-7163-4db7-a12b-1d0dc6177537)

There’re few interesting observations: I never knew you could rent a boat as accommodation on AirBnB and there’re 8 listings among the ‘best’ listings. House property type and villa is cheaper on average than an apartment. The cheapest option is a dorm with a price of $50 and only 1 listing. Let’s visualize this:

![Prices per property type](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/b2ff83ff-5462-49d5-b448-9e36ec2cd7f9)

If it were up to me, I would pick an apartment to stay even though boat seems like a fun option. Let’s check apartments distribution by the neighborhood.

```
# Filter the data for 'apartment' property_type
apartment_data = best[best['property_type'] == 'Apartment']

neighborhood_counts = apartment_data['neighbourhood'].value_counts()
mean_price = apartment_data.groupby('neighbourhood')['price'].mean()

# Create a bar chart
plt.figure(figsize=(10, 8))
neighborhood_counts.plot(kind='bar')
plt.title('# of Best Apartments Per Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('# of Apartments')
plt.xticks(rotation=90)  # Rotate the neighborhood names on x-axis by 90 degrees
plt.tight_layout()
plt.show()
```
![Distribution of best apartments per neighborhood](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/40ace3f5-97fd-4a55-8ded-f3d7302f1573)

South End seems to be a popular neighborhood for AirBnB realtors/owners. There are 5 neighborhoods with 100+ listings in each with ‘best’ apartments. I’d probably narrow my searches there. Let’s add average prices to the graph and reevaluate.

```
apartment_counts = apartment_data['neighbourhood'].value_counts()
average_prices = apartment_data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)
fig, ax1 = plt.subplots(figsize=(10, 8))  # Increased figure size for better distinction

# Bar chart for the number of apartments
color = 'tab:blue'
ax1.set_xlabel('Neighbourhood')
ax1.set_ylabel('Number of Apartments', color=color)
ax1.bar(average_prices.index, apartment_counts.reindex(average_prices.index).values, color=color, width=0.4, align='center')
ax1.tick_params(axis='y', labelcolor=color)

ax1.set_xticklabels(average_prices.index, rotation=90)

ax2 = ax1.twinx()  

# Bar chart for the average price
color = 'tab:orange' 
ax2.set_ylabel('Average Price', color=color)
ax2.bar(average_prices.index, average_prices.values, color=color, width=0.4, align='edge')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
plt.show()

```

![Price per neighborhood](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/6bfa44b3-03db-4715-a77e-72be016366f9)

There are few insights we can see: Financial District is the most expensive neighborhood on average but has few listings available. South End has the highest count of listings but the price is a bit above average. I would personally check the listings in Jamaica Plain and Allston-Brighton as on average they’re relatively cheap, there many options with good reviews. 

### Predicting the price using linear regression. 

Here I decided to limit the features to number of bedrooms, bathrooms, property type and room type. I believe if you aren’t aware what neighborhoods are classified as good or bad, these variables should be a great indicator of what a price should be. 

```
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float).astype('int64')
listings = listings.dropna(subset=['property_type', 'room_type','bathrooms', 'bedrooms'])

# Selecting the features and target variable
X = listings[['bedrooms', 'bathrooms', 'property_type', 'room_type']]
y = listings['price']

# One-hot encoding the categorical variables
categorical_features = ['property_type', 'room_type']
one_hot_encoder = OneHotEncoder()

# Using ColumnTransformer to apply the encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initializing the Random Forest Regressor
lr = LinearRegression()

# Fitting the model
lr.fit(X_train, y_train)

# Making predictions
y_pred = lr.predict(X_test)

# Calculating the Mean Squared Error
"The r-squared score for the model was {} on {} values.".format(r2_score(y_test, y_pred), len(y_test))
```

The r-squared score is 0.42. The chosen variables predict only 42% of price variance. The performance of the model is not the best. This being said, there’re only 712 values in y_test, so maybe if we had a larger dataset, model would have performed better. It can also be improved by tuning predicting variables.

Thanks for reading this far! I wish you a wonderful day!









