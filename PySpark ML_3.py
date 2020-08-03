# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:09:19 2020

@author: RikSo
"""
"""
Encoding flight origin
The org column in the flights data is a categorical variable giving the airport from which a flight departs.

ORD — O'Hare International Airport (Chicago)
SFO — San Francisco International Airport
JFK — John F Kennedy International Airport (New York)
LGA — La Guardia Airport (New York)
SMF — Sacramento
SJC — San Jose
TUS — Tucson International Airport
OGG — Kahului (Hawaii)
Obviously this is only a small subset of airports. Nevertheless, since this is a categorical variable, it needs to be one-hot encoded before it can be used in a regression model.

The data are in a variable called flights. You have already used a string indexer to create a column of indexed values corresponding to the strings in org.

Note:: You might find it useful to revise the slides from the lessons in the Slides panel next to the IPython Shell.
"""

"""
Subset from the flights DataFrame:

+---+-------+
|org|org_idx|
+---+-------+
|JFK|2.0    |
|ORD|0.0    |
|SFO|1.0    |
|ORD|0.0    |
|ORD|0.0    |
+---+-------+
only showing top 5 rows
"""

# Import the one hot encoder class
from pyspark.ml.feature import OneHotEncoderEstimator

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)

# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()


"""
Generate a summary of the mapping from categorical values to binary encoded dummy variables. 
Include only unique values and order by org_idx
+---+-------+-------------+
|org|org_idx|    org_dummy|
+---+-------+-------------+
|ORD|    0.0|(7,[0],[1.0])|
|SFO|    1.0|(7,[1],[1.0])|
|JFK|    2.0|(7,[2],[1.0])|
|LGA|    3.0|(7,[3],[1.0])|
|SJC|    4.0|(7,[4],[1.0])|
|SMF|    5.0|(7,[5],[1.0])|
|TUS|    6.0|(7,[6],[1.0])|
|OGG|    7.0|    (7,[],[])|
+---+-------+-------------+
"""

"""
Flight duration model: Just distance
In this exercise you'll build a regression model to predict flight duration (the duration column).

For the moment you'll keep the model simple, including only the distance of the flight 
(the km column) as a predictor.

The data are in flights. The first few records are displayed in the terminal. 
These data have also been split into training and testing sets and are available 
as flights_train and flights_test.

Subset from the flights DataFrame:

+------+--------+--------+
|km    |features|duration|
+------+--------+--------+
|3465.0|[3465.0]|351     |
|509.0 |[509.0] |82      |
|542.0 |[542.0] |82      |
|1989.0|[1989.0]|195     |
|415.0 |[415.0] |65      |
+------+--------+--------+
only showing top 5 rows
"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol='duration').evaluate(predictions)

"""
+--------+------------------+
    |duration|prediction        |
    +--------+------------------+
    |105     |118.71205377865795|
    |204     |174.69339409767792|
    |160     |152.16523695718402|
    |297     |337.8153345965721 |
    |105     |113.5132482846978 |
    +--------+------------------+
    only showing top 5 rows
"""

"""
Interpreting the coefficients
The linear regression model for flight duration as a function of distance takes the form

duration=α+β×distance
where

α — intercept (component of duration which does not depend on distance) and
β — coefficient (rate at which duration increases as a function of distance; also called the slope).
By looking at the coefficients of your model you will be able to infer

how much of the average flight duration is actually spent on the ground and
what the average speed is during a flight.
The linear regression model is available as regression.
"""

# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)

# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)

# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)

"""
44.36345473899361
[0.07566671399881963]
0.07566671399881963
792.9510458315392
"""


"""
Flight duration model: Adding origin airport
Some airports are busier than others. Some airports are bigger than others too. 
Flights departing from large or busy airports are likely to spend more time taxiing or 
waiting for their takeoff slot. 
So it stands to reason that the duration of a flight might depend not only on the distance 
being covered but also the airport from which the flight departs.

You are going to make the regression model a little more sophisticated 
by including the departure airport as a predictor.

These data have been split into training and testing sets and are available as flights_train 
and flights_test. The origin airport, stored in the org column, has been indexed into org_idx, 
which in turn has been one-hot encoded into org_dummy. 
The first few records are displayed in the terminal.


Subset from the flights DataFrame:

+------+-------+-------------+----------------------+
|km    |org_idx|org_dummy    |features              |
+------+-------+-------------+----------------------+
|3465.0|2.0    |(7,[2],[1.0])|(8,[0,3],[3465.0,1.0])|
|509.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[509.0,1.0]) |
|542.0 |1.0    |(7,[1],[1.0])|(8,[0,2],[542.0,1.0]) |
|1989.0|0.0    |(7,[0],[1.0])|(8,[0,1],[1989.0,1.0])|
|415.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[415.0,1.0]) |
+------+-------+-------------+----------------------+
only showing top 5 rows
"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
RegressionEvaluator(labelCol='duration').evaluate(predictions)

"""
11.122486328678256
"""

"""
Interpreting coefficients
Remember that origin airport, org, has eight possible values 
(ORD, SFO, JFK, LGA, SMF, SJC, TUS and OGG) which have been one-hot encoded 
to seven dummy variables in org_dummy.

The values for km and org_dummy have been assembled into features, 
which has eight columns with sparse representation. Column indices in features are as follows:

0 — km
1 — ORD
2 — SFO
3 — JFK
4 — LGA
5 — SMF
6 — SJC and
7 — TUS.
Note that OGG does not appear in this list 
because it is the reference level for the origin airport category.

In this exercise you'll be using the intercept and coefficients attributes to interpret the model.

The coefficients attribute is a list, 
where the first element indicates how flight duration changes with flight distance.
"""

# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)

"""
You're going to spend over an hour on the ground at JFK or LGA but only around 15 minutes at OGG.
807.3336599681242
15.856628374450773
68.53550999587868
62.56747182033072
"""

"""
Bucketing departure time
Time of day data are a challenge with regression models. 
They are also a great candidate for bucketing.

In this lesson you will convert the flight departure times 
from numeric values between 0 (corresponding to 00:00) and 24 (corresponding to 24:00) 
to binned values. You'll then take those binned values and one-hot encode them.

"""
from pyspark.ml.feature import Bucketizer, OneHotEncoderEstimator

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0, 3, 6, 9, 12, 15, 18, 21, 24], inputCol='depart', outputCol='depart_bucket')

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)
"""
+------+-------------+
|depart|depart_bucket|
+------+-------------+
|  9.48|          3.0|
| 16.33|          5.0|
|  6.17|          2.0|
| 10.33|          3.0|
|  8.92|          2.0|
+------+-------------+
only showing top 5 rows
"""

# Create a one-hot encoder
onehot = OneHotEncoderEstimator(inputCols=['depart_bucket'], outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
"""
Train the encoder on the data and then use it to convert 
the bucketed departure times to dummy variables. Show the first five values 
for depart, depart_bucket and depart_dummy.
"""
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)

"""
+------+-------------+-------------+
|depart|depart_bucket| depart_dummy|
+------+-------------+-------------+
|  9.48|          3.0|(7,[3],[1.0])|
| 16.33|          5.0|(7,[5],[1.0])|
|  6.17|          2.0|(7,[2],[1.0])|
| 10.33|          3.0|(7,[3],[1.0])|
|  8.92|          2.0|(7,[2],[1.0])|
+------+-------------+-------------+
only showing top 5 rows
"""

"""
Flight duration model: Adding departure time
In the previous exercise the departure time was bucketed and converted to dummy variables. 
Now you're going to include those dummy variables in a regression model for flight duration.

The data are in flights. The km, org_dummy and depart_dummy columns have been assembled 
into features, where km is index 0, org_dummy runs from index 1 to 7 and depart_dummy 
from index 8 to 14.

The data have been split into training and testing sets and a linear regression model, 
regression, has been built on the training data. Predictions have been made on 
the testing data and are available as predictions
"""
"""
Feature columns:

 0 — km
 1 — ORD
 2 — SFO
 3 — JFK
 4 — LGA
 5 — SJC
 6 — SMF
 7 — TUS
 8 — 00:00 to 03:00
 9 — 03:00 to 06:00
10 — 06:00 to 09:00
11 — 09:00 to 12:00
12 — 12:00 to 15:00
13 — 15:00 to 18:00
14 — 18:00 to 21:00
"""
# Find the RMSE on testing data
from pyspark.ml.evaluation import RegressionEvaluator
RegressionEvaluator(labelCol='duration').evaluate(predictions)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 00:00 and 03:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 00:00 and 03:00
avg_night_jfk = regression.intercept + regression.coefficients[3] + regression.coefficients[8]
print(avg_night_jfk)

"""
10.475615792093903
-4.125122945654926
47.580713975630594
"""


"""
Flight duration model: More features!
Let's add more features to our model. This will not necessarily result in a better model. Adding some features might improve the model. Adding other features might make it worse.

More features will always make the model more complicated and difficult to interpret.

These are the features you'll include in the next model:

km
org (origin airport, one-hot encoded, 8 levels)
depart (departure time, binned in 3 hour intervals, one-hot encoded, 8 levels)
dow (departure day of week, one-hot encoded, 7 levels) and
mon (departure month, one-hot encoded, 12 levels).
These have been assembled into the features column, which is a sparse representation of 32 columns (remember one-hot encoding produces a number of columns which is one fewer than the number of levels).

The data are available as flights, randomly split into flights_train and flights_test. The object predictions is also available.

Subset from the flights DataFrame:

+--------------------------------------------+--------+
|features                                    |duration|
+--------------------------------------------+--------+
|(32,[0,3,11],[3465.0,1.0,1.0])              |351     |
|(32,[0,1,13,17,21],[509.0,1.0,1.0,1.0,1.0]) |82      |
|(32,[0,2,10,19,23],[542.0,1.0,1.0,1.0,1.0]) |82      |
|(32,[0,1,11,16,30],[1989.0,1.0,1.0,1.0,1.0])|195     |
|(32,[0,1,10,20,25],[415.0,1.0,1.0,1.0,1.0]) |65      |
+--------------------------------------------+--------+
only showing top 5 rows

"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit linear regression model to training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)
"""
The test RMSE is 9.932871937636765
    [0.07466666934355301,8.078881563529334,1.9653119764127,30.90058109033576,
    25.975255321620892,-4.673220904608639,-0.5071146737729627,-19.35458238364859,
    -20.553418861596572,-1.5250162109404266,2.902212223024328,6.9230292833642215,
    4.830665365251793,7.567402763210368,6.7482370748914215,0.5888385584814597,
    0.9652580580230514,-0.5645623742771357,-0.6200119406154614,-1.2240717932722625,
    -1.3845856065896651,-4.574194011951068,-6.482639010679108,-3.8632479519852603,
    -3.7540130731837587,-8.8096609834927,-6.500070642930037,-5.396616986276698,
    -5.1580203920599885,-9.682260059912322,-5.6441219946379695,-5.467775936528763]
"""

"""
Flight duration model: Regularisation!
In the previous exercise you added more predictors to the flight duration model. 
The model performed well on testing data, but with so many coefficients it was difficult 
to interpret.

In this exercise you'll use Lasso regression (regularized with a L1 penalty) 
to create a more parsimonious model. Many of the coefficients in the resulting model 
will be set to zero. This means that only a subset of the predictors actually contribute 
to the model. Despite the simpler model, it still produces a good RMSE on the testing data.

You'll use a specific value for the regularization strength. 
Later you'll learn how to find the best value using cross validation.

The data (same as previous exercise) are available as flights, 
randomly split into flights_train and flights_test.
"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit Lasso model (α = 1) to training data
regression = LinearRegression(labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(regression.transform(flights_test))
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)

"""
The test RMSE is 11.221618112066176
[0.07326284332459325,0.26927242574175647,-4.213823507520847,23.31411303902282,
16.924833465407964,-7.538366699625629,-5.04321753247765,-20.348693139176927,
0.0,0.0,0.0,0.0,0.0,1.199161974782719,0.43548357163388335,0.0,0.0,0.0,0.0,0.
0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
Number of coefficients equal to 0: 22
"""