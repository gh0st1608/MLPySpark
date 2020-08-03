# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:53:09 2020

@author: RikSo
"""

"""
Removing columns and rows
You previously loaded airline flight data from a CSV file. You're going to develop a model 
which will predict whether or not a given flight will be delayed.

In this exercise you need to trim those data down by:

removing an uninformative column and
removing rows which do not have information about whether or not a flight was delayed.

Note:: You might find it useful to revise the slides from the lessons in the Slides panel
next to the IPython Shell.
"""

# Remove the 'flight' column
flights_drop_column = flights.drop('flight')

# Number of records with missing 'delay' values
flights_drop_column.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())

"""
Column manipulation
The Federal Aviation Administration (FAA) considers a flight to be "delayed" when it arrives 15 minutes or more after its scheduled time.

The next step of preparing the flight data has two parts:

convert the units of distance, replacing the mile column with a kmcolumn; and
create a Boolean column indicating whether or not a flight was delayed.

Import a function which will allow you to round a number to a specific number of decimal places.
Derive a new km column from the mile column, rounding to zero decimal places. 
One mile is 1,60934 km.

Remove the mile column.
Create a label column with a value of 1 indicating the delay was 15 minutes 
or more and 0 otherwise.

"""

# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)


"""
Categorical columns
In the flights data there are two columns, carrier and org, which hold categorical data. 
You need to transform those columns into indexed numerical values.

Instructions:
Import the appropriate class and create an indexer object to transform the carrier column from a string to an numeric index.
Prepare the indexer object on the flight data.
Use the prepared indexer to create the numeric index column.
Repeat the process for the org column.

"""

from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)

"""
Assembling columns
The final stage of data preparation is to consolidate all of the predictor columns into a single column.

At present our data has the following predictor columns:

mon, dom and dow
carrier_idx (derived from carrier)
org_idx (derived from org)
km
depart
duration
"""


# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=[
    'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'
], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)

"""
+-----------------------------------------+-----+
    |features                                 |delay|
    +-----------------------------------------+-----+
    |[0.0,22.0,2.0,0.0,0.0,509.0,16.33,82.0]  |30   |
    |[2.0,20.0,4.0,0.0,1.0,542.0,6.17,82.0]   |-8   |
    |[9.0,13.0,1.0,1.0,0.0,1989.0,10.33,195.0]|-5   |
    |[5.0,2.0,1.0,0.0,1.0,885.0,7.98,102.0]   |2    |
    |[7.0,2.0,6.0,1.0,0.0,1180.0,10.83,135.0] |54   |
    +-----------------------------------------+-----+
    only showing top 5 rows
"""


"""
Train/test split
To objectively assess a Machine Learning model you need to be able to test it on an independent set of data. You can't use the same data that you used to train the model: of course the model will perform (relatively) well on those data!

You will split the data into two components:

training data (used to train the model) and
testing data (used to test the model).
"""

# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2],seed=17)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights_test.count()
print(training_ratio)
"""
3.8886576482830386
"""

"""
Build a Decision Tree
Now that you've split the flights data into training and testing sets, you can use the training set to fit a Decision Tree model.

The data are available as flights_train and flights_test.

NOTE: It will take a few seconds for the model to train... please be patient!
"""
# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)
"""
-----+----------+----------------------------------------+
|label|prediction|probability                             |
+-----+----------+----------------------------------------+
|1    |1.0       |[0.2911010558069382,0.7088989441930619] |
|1    |1.0       |[0.3875,0.6125]                         |
|1    |1.0       |[0.3875,0.6125]                         |
|0    |0.0       |[0.6337448559670782,0.3662551440329218] |
|0    |0.0       |[0.9368421052631579,0.06315789473684211]|
+-----+----------+----------------------------------------+
only showing top 5 rows
"""

"""
Evaluate the Decision Tree
You can assess the quality of your model by evaluating how well it performs on the testing data. Because the model was not trained on these data, this represents an objective assessment of the model.

A confusion matrix gives a useful breakdown of predictions versus known values. It has four cells which represent the counts of:

True Negatives (TN) — model predicts negative outcome & known outcome is negative
True Positives (TP) — model predicts positive outcome & known outcome is positive
False Negatives (FN) — model predicts negative outcome but known outcome is positive
False Positives (FP) — model predicts positive outcome but known outcome is negative.
"""

# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label <> prediction').count()
FP = prediction.filter('prediction = 1 AND label <> prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP)/(TN + TP + FN + FP)
print(accuracy)

"""
+-----+----------+-----+
|    1|       0.0|  154|
|    0|       0.0|  289|
|    1|       1.0|  328|
|    0|       1.0|  190|
+-----+----------+-----+

0.6420395421436004
"""

"""
Build a Logistic Regression model
You've already built a Decision Tree model using the flights data. Now you're going to create a Logistic Regression model on the same data.

The objective is to predict whether a flight is likely to be delayed by at least 15 minutes (label 1) or not (label 0).

Although you have a variety of predictors at your disposal, you'll only use the mon, depart and duration columns for the moment. These are numerical features which can immediately be used for a Logistic Regression model. You'll need to do a little more work before you can include categorical features. Stay tuned!

The data have been split into training and testing sets and are available as flights_train and flights_test
"""

# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy('label', 'prediction').count().show()

"""
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|    1|       0.0|  195|
|    0|       0.0|  288|
|    1|       1.0|  277|
|    0|       1.0|  201|
+-----+----------+-----+
"""

"""
First few rows from the flights DataFrame:

+---+------+--------+-----------------+-----+
|mon|depart|duration|features         |label|
+---+------+--------+-----------------+-----+
|0  |16.33 |82      |[0.0,16.33,82.0] |1    |
|2  |6.17  |82      |[2.0,6.17,82.0]  |0    |
|9  |10.33 |195     |[9.0,10.33,195.0]|0    |
|5  |7.98  |102     |[5.0,7.98,102.0] |0    |
|7  |10.83 |135     |[7.0,10.83,135.0]|1    |
+---+------+--------+-----------------+-----+
only showing top 5 rows
"""

"""
Evaluate the Logistic Regression model
Accuracy is generally not a very reliable metric because it can be biased by the most common target class.

There are two other useful metrics:

precision and
recall.
Check the slides for this lesson to get the relevant expressions.

Precision is the proportion of positive predictions which are correct. For all flights which are predicted to be delayed, what proportion is actually delayed?

Recall is the proportion of positives outcomes which are correctly predicted. For all delayed flights, what proportion is correctly predicted by the model?

The precision and recall are generally formulated in terms of the positive target class. But it's also possible to calculate weighted versions of these metrics which look at both target classes.

The components of the confusion matrix are available as TN, TP, FN and FP, as well as the object prediction.
"""

"""
First few predictions from the Logistic Regression model:

+-----+----------+----------------------------------------+
|label|prediction|probability                             |
+-----+----------+----------------------------------------+
|0    |1.0       |[0.48618640716970973,0.5138135928302903]|
|1    |0.0       |[0.52242444215606,0.47757555784394007]  |
|0    |0.0       |[0.5726551829113304,0.4273448170886696] |
|0    |0.0       |[0.5149292596494213,0.4850707403505788] |
|1    |0.0       |[0.5426764281965827,0.4573235718034173] |
+-----+----------+----------------------------------------+
only showing top 5 rows
"""

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})

"""
precision = 0.58
recall    = 0.59
"""

"""
Punctuation, numbers and tokens
At the end of the previous chapter you loaded a dataset of SMS messages which had been labeled as either "spam" (label 1) or "ham" (label 0). You're now going to use those data to build a classifier model.

But first you'll need to prepare the SMS messages as follows:

remove punctuation and numbers
tokenize (split into individual words)
remove stop words
apply the hashing trick
convert to TF-IDF representation.
In this exercise you'll remove punctuation and numbers, then tokenize the messages.

The SMS data are available as sms.
"""

"""
+---+-------------------------------------------+-----+
|id |text                                       |label|
+---+-------------------------------------------+-----+
|1  |Sorry, I'll call later in meeting          |0    |
|2  |Dont worry. I guess he's busy.             |0    |
|3  |Call FREEPHONE 0800 542 0578 now!          |1    |
|4  |Win a 1000 cash prize or a prize worth 5000|1    |
+---+-------------------------------------------+-----+
only showing top 4 rows
"""


# Import the necessary functions
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol='words').transform(wrangled)

wrangled.show(4, truncate=False)

"""
+---+----------------------------------+-----+------------------------------------------+
    |id |text                              |label|words                                     |
    +---+----------------------------------+-----+------------------------------------------+
    |1  |Sorry I'll call later in meeting  |0    |[sorry, i'll, call, later, in, meeting]   |
    |2  |Dont worry I guess he's busy      |0    |[dont, worry, i, guess, he's, busy]       |
    |3  |Call FREEPHONE now                |1    |[call, freephone, now]                    |
    |4  |Win a cash prize or a prize worth |1    |[win, a, cash, prize, or, a, prize, worth]|
    +---+----------------------------------+-----+------------------------------------------+
    only showing top 4 rows
"""


"""
Stop words and hashing
The next steps will be to remove stop words and then apply the hashing trick, converting the results into a TF-IDF.

A quick reminder about these concepts:

The hashing trick provides a fast and space-efficient way to map a very large (possibly infinite) set of items (in this case, all words contained in the SMS messages) onto a smaller, finite number of values.
The TF-IDF matrix reflects how important a word is to each document. It takes into account both the frequency of the word within each document but also the frequency of the word across all of the documents in the collection.
The tokenized SMS data are stored in sms in a column named words. You've cleaned up the handling of spaces in the data so that the tokenized text is neater.


El truco de hashing proporciona una manera rápida y eficiente en el espacio para mapear un conjunto muy grande (posiblemente infinito) de elementos (en este caso, todas las palabras contenidas en los mensajes SMS) en un número de valores más pequeño y finito.
La matriz TF-IDF refleja la importancia de una palabra para cada documento. Tiene en cuenta tanto la frecuencia de la palabra dentro de cada documento como la frecuencia de la palabra en todos los documentos de la colección.
Los datos de SMS tokenizados se almacenan en sms en una columna llamada palabras. Ha limpiado el manejo de espacios en los datos para que el texto tokenizado sea más ordenado.
"""

"""
First few rows from the sms DataFrame:

+---+---------------------------------------------------------------------------------------------------------------------------+-----+
|id |words                                                                                                                      |label|
+---+---------------------------------------------------------------------------------------------------------------------------+-----+
|1  |[sorry, i'll, call, later, in, meeting]                                                                                    |0    |
|2  |[dont, worry, i, guess, he's, busy]                                                                                        |0    |
|3  |[call, freephone, now]                                                                                                     |1    |
|4  |[win, a, cash, prize, or, a, prize, worth]                                                                                 |1    |
|5  |[go, until, jurong, point, crazy, available, only, in, bugis, n, great, world, la, e, buffet, cine, there, got, amore, wat]|0    |
+---+---------------------------------------------------------------------------------------------------------------------------+-----+
only showing top 5 rows
"""

from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF

# Remove stop words.
wrangled = StopWordsRemover(inputCol='words', outputCol='terms')\
      .transform(sms)

# Apply the hashing trick
wrangled = HashingTF(inputCol='terms', outputCol='hash', numFeatures=1024)\
      .transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol='hash', outputCol='features')\
      .fit(wrangled).transform(wrangled)
      
tf_idf.select('terms', 'features').show(4, truncate=False)

"""
 +--------------------------------+----------------------------------------------------------------------------------------------------+
    |terms                           |features                                                                                            |
    +--------------------------------+----------------------------------------------------------------------------------------------------+
    |[sorry, call, later, meeting]   |(1024,[138,344,378,1006],[2.2391682769656747,2.892706319430574,3.684405173719015,4.244020961654438])|
    |[dont, worry, guess, busy]      |(1024,[53,233,329,858],[4.618714411095849,3.557143394108088,4.618714411095849,4.937168142214383])   |
    |[call, freephone]               |(1024,[138,396],[2.2391682769656747,3.3843005812686773])                                            |
    |[win, cash, prize, prize, worth]|(1024,[31,69,387,428],[3.7897656893768414,7.284881949239966,4.4671645129686475,3.898659777615979])  |
    +--------------------------------+----------------------------------------------------------------------------------------------------+
    only showing top 4 rows
"""


"""
Training a spam classifier
The SMS data have now been prepared for building a classifier. Specifically, this is what you have done:

removed numbers and punctuation
split the messages into words (or "tokens")
removed stop words
applied the hashing trick and
converted to a TF-IDF representation.
Next you'll need to split the TF-IDF data into training and testing sets. Then you'll use the training data to fit a Logistic Regression model and finally evaluate the performance of that model on the testing data.

The data are stored in sms and LogisticRegression has been imported for you.

Selected columns from first few rows of the sms DataFrame:

+-----+--------------------+
|label|            features|
+-----+--------------------+
|    0|(1024,[138,344,37...|
|    0|(1024,[53,233,329...|
|    1|(1024,[138,396],[...|
|    1|(1024,[31,69,387,...|
|    0|(1024,[116,262,33...|
+-----+--------------------+
only showing top 5 rows
"""

# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([0.8,0.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()

"""
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|    1|       0.0|   47|
|    0|       0.0|  987|
|    1|       1.0|  124|
|    0|       1.0|    3|
+-----+----------+-----+
"""

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])
print(pipeline)
"""Pipeline_2721762009b9"""