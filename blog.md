# An attempt of analyzing Boston AirBnB dataset has been made. Only one way to see if it's any good.

## Udacity project: Writing a Data Scientist Blog Post

### This project is related to Boston AirBnB dataset and deliverables of the project is to come up with at least 3 business questions to be answered by data analysis. I've approached this dataset with questions I would have if I were planning a trip to Boston and was considering using AirBnB instead of a hotel. 

Questions I'm trying to answer by this analysis:

1.	How are prices affected by seasonality? Would the summer months be the most expensive ones as they tend to? What about days of the week – are weekends the most expensive period?
2.	What are the best properties to consider booking? 
3.	By looking at variables provided in the listing dataset, I believe number of rooms, bathrooms, property type and room type would be a best set of variables to predict the price of a listing. Would this be the case in a linear regression model?



### To answer first question let’s group up the average price of listings per month.

![Price per month](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/ab6c83d3-41ae-4a5b-815d-c65665dd2a17)


Turns out my assumption was wrong. Period from August to November is the most expensive one with September being top 1. Interesting to see that November has higher prices than July or June. I'm not a fan of colder months, so if I were to visit I'd choose May-June.

Now let’s check prices for days of the week.

![Price per day of week](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/326a1507-041c-46b7-94f1-759f64e13efc)


As expected, weekends are the most expensive. This being said the difference is not that high – only $8 between the highest and lowest prices. If I lived close to Boston area and wanted to visit the city, going there on weekends still might be a good idea as saving $8 per day should be worth the conveience of travelling on the weekend.

### Now, onto the next question - What are the best properties to consider booking? 

If you’re anything like me, you’d go for listings with highest scores and you’d look for the average price within this best category. I’d define it here using the overall rating – those with 90 or higher. Let’s start by inspecting average price per type of property and their count. 

![best_group_by](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/637d7fdd-7163-4db7-a12b-1d0dc6177537)

There’re few interesting observations: I never knew you could rent a boat as accommodation on AirBnB and there’re 8 listings among the ‘best’ listings. House property type and villa is cheaper on average than an apartment. The cheapest option is a dorm with a price of $50 and only 1 listing. Let’s visualize this:

![Prices per property type](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/b2ff83ff-5462-49d5-b448-9e36ec2cd7f9)

If it were up to me, I would pick an apartment to stay even though boat seems like a fun option. Let’s check apartments distribution by the neighborhood.

![Distribution of best apartments per neighborhood](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/40ace3f5-97fd-4a55-8ded-f3d7302f1573)

South End seems to be a popular neighborhood for AirBnB realtors/owners. There are 5 neighborhoods with 100+ listings in each with ‘best’ apartments. I’d probably narrow my searches there. Let’s add average prices to the graph and reevaluate.

![Price per neighborhood](https://github.com/bakewka94/Udacity---BostonAirBnB/assets/70720851/6bfa44b3-03db-4715-a77e-72be016366f9)

There are few insights we can see: Financial District is the most expensive neighborhood on average but has few listings available. South End has the highest count of listings but the price is a bit above average. I would personally check the listings in Jamaica Plain and Allston-Brighton as on average they’re relatively cheap, there many options with good reviews. 

### Predicting the price using linear regression. 

Here I decided to limit the features to number of bedrooms, bathrooms, property type and room type. I believe if you aren’t aware what neighborhoods are classified as good or bad, these variables should be a great indicator of what a price should be. 

The r-squared score is 0.42. The chosen variables predict only 42% of price variance. The performance of the model is not the best to put it lightly. This being said, there’re only 712 values in y_test, so maybe if we had a larger dataset, model would have performed better. It can also be improved by tuning predicting variables.

Thanks for reading this far! I wish you a wonderful day!
