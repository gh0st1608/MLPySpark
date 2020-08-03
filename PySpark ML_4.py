# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:21:39 2020

@author: RikSo
"""
"""
Flight duration model: Pipeline stages
You're going to create the stages for the flights duration model pipeline. 
You will use these in the next exercise to build a pipeline and to create a regression model.
"""

# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoderEstimator(inputCols=['org_idx','dow'], outputCols=['org_dummy','dow_dummy'])

# Assemble predictors into a single column
assembler = VectorAssembler(inputCols=['km','org_dummy','dow_dummy'], outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')


"""
Flight duration model: Pipeline model
You're now ready to put those stages together in a pipeline.

You'll construct the pipeline and then train the pipeline on the training data. 
This will apply each of the individual stages in the pipeline to the training data in turn. 
None of the stages will be exposed to the testing data at all: there will be no leakage!

Once the entire pipeline has been trained it will then be used to make predictions 
on the testing data.

The data are available as flights, which has been randomly split into flights_train 
and flights_test.
"""

# Import class for creating a pipeline
from pyspark.ml import Pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer,onehot,assembler,regression])

# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)
"""
print(predictions)
DataFrame[mon: int, dom: int, dow: int, carrier: string, flight: int, 
          org: string, depart: double, duration: int, delay: int, 
          km: double, org_idx: double, org_dummy: vector, dow_dummy: vector, 
          features: vector, prediction: double]

print(pipeline)
PipelineModel_ce5863305799

"""



"""
SMS spam pipeline
You haven't looked at the SMS data for quite a while. Last time we did the following:

*split the text into tokens
*removed stop words
*applied the hashing trick
*converted the data from counts to IDF and
*trained a linear regression model.

Each of these steps was done independently. This seems like a great application for a pipeline!
"""
"""
Selected columns from first few rows of the sms DataFrame:

+---+---------------------------------+-----+
|id |text                             |label|
+---+---------------------------------+-----+
|1  |Sorry I'll call later in meeting |0    |
|2  |Dont worry I guess he's busy     |0    |
|3  |Call FREEPHONE now               |1    |
|4  |Win a cash prize or a prize worth|1    |
+---+---------------------------------+-----+
only showing top 4 rows
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


"""
Cross validating simple flight duration model
You've already built a few models for predicting flight duration and evaluated 
them with a simple train/test split. However, 
cross-validation provides a much better way to evaluate model performance.

In this exercise you're going to train a simple model for flight duration using cross-validation. 
Travel time is usually strongly correlated with distance, 
so using the km column alone should give a decent model.

The data have been randomly split into flights_train and flights_test.

The following classes have already been imported: LinearRegression, 
RegressionEvaluator, ParamGridBuilder and CrossValidator.

Subset from the flights DataFrame:

+------+--------+--------+
|km    |features|duration|
+------+--------+--------+
|542.0 |[542.0] |82      |
|885.0 |[885.0] |102     |
|2317.0|[2317.0]|232     |
|2943.0|[2943.0]|250     |
|1765.0|[1765.0]|190     |
+------+--------+--------+
only showing top 5 rows
"""
# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol='duration')
evaluator = RegressionEvaluator(labelCol='duration')

# Create a cross validator
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, the fit() method can take a little while to complete.

"""
Cross validating flight duration model pipeline
The cross-validated model that you just built was simple, using km alone to predict duration.

Another important predictor of flight duration is the origin airport. Flights generally take longer to get into the air from busy airports. Let's see if adding this predictor improves the model!

In this exercise you'll add the org field to the model. However, since org is categorical, there's more work to be done before it can be included: it must first be transformed to an index and then one-hot encoded before being assembled with km and used to build the regression model. We'll wrap these operations up in a pipeline.

The following objects have already been created:

params — an empty parameter grid
evaluator — a regression evaluator
regression — a LinearRegression object with labelCol='duration'.
All of the required classes have already been imported.
"""

# Create an indexer for the org field
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=['km','org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
          estimatorParamMaps=params,
          evaluator=evaluator)



"""
Optimizing flights linear regression
Up until now you've been using the default hyper-parameters when building your models. 
In this exercise you'll use cross validation to choose an optimal (or close to optimal) 
set of model hyper-parameters.

The following have already been created:

regression — a LinearRegression object
pipeline — a pipeline with string indexer, one-hot encoder, vector assembler 
and linear regression and
evaluator — a RegressionEvaluator object.
"""

# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01,0.1,1.00,10]) \
               .addGrid(regression.elasticNetParam, [0,0.5,1])

# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params)) """12 Modelos"""

# Create cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)


"""
Dissecting the best flight duration model
You just set up a CrossValidator to find good parameters for the linear regression model predicting flight duration.

Now you're going to take a closer look at the resulting model, split out the stages and use it to make predictions on the testing data.

The following have already been created:

cv — a trained CrossValidatorModel object and
evaluator — a RegressionEvaluator object.
The flights data have been randomly split into flights_train and flights_test.
"""

"""
The default model has RMSE of 10.614372 on testing data.
"""

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)
"""
[StringIndexer_14299b2d5472, OneHotEncoderEstimator_9a650c117f1d, VectorAssembler_933acae88a6e, LinearRegression_9f5a93965597]
"""

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()


# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
print(predictions)
"""
DataFrame[mon: int, dom: int, dow: int, carrier: string, flight: int, 
org: string, depart: double, duration: int, delay: int, km: double, 
org_idx: double, org_dummy: vector, features: vector, prediction: double]
"""

evaluator.evaluate(predictions)
print(evaluator.evaluate(predictions))
"""10.516377654959923"""


"""
SMS spam optimised
The pipeline you built earlier for the SMS spam model used the default parameters for all of the elements in the pipeline. It's very unlikely that these parameters will give a particularly good model though.

In this exercise you'll set up a parameter grid which can be used with cross validation to choose a good set of parameters for the SMS spam classifier.

The following are already defined:

hasher — a HashingTF object and
logistic — a LogisticRegression object.
"""
# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024,4096,16384]) \
               .addGrid(hasher.binary, [True,False])

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01,0.1,1,10]) \
               .addGrid(logistic.elasticNetParam, [0,0.5,1.0])

# Build parameter grid
params = params.build()



"""
How many models for grid search?
How many models will be built when the cross-validator below is fit to data?

params = ParamGridBuilder().addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
                           .addGrid(hasher.binary, [True, False]) \
                           .addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]) \
                           .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0]) \
                           .build()

cv = CrossValidator(..., estimatorParamMaps=params, numFolds=5)


72 is the number of points in the parameter grid, but you need to also take 
into account the number of models built during cross-validation.
360 is the number of models built = 72 * numFolds

There are 72 points in the parameter grid and 5 folds in the cross-validator. 
The product is 360. It takes time to build all of those models, 
which is why we're not doing it here!
"""

"""
Delayed flights with Gradient-Boosted Trees
You've previously built a classifier for flights likely to be delayed using a Decision Tree. 
In this exercise you'll compare a Decision Tree model to a Gradient-Boosted Trees model.

The flights data have been randomly split into flights_train and flights_test.
"""

Subset of data from the flights DataFrame:

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

from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(tree.transform(flights_test))
evaluator.evaluate(gbt.transform(flights_test))

# Find the number of trees and the relative importance of features
print(gbt.getNumTrees)
"""20"""

print(gbt.featureImportances)
"""(3,[0,1,2],[0.2890040527329095,0.3013523658017735,0.40964358146531693])"""

"""
Delayed flights with a Random Forest
In this exercise you'll bring together cross validation and ensemble methods. 
You'll be training a Random Forest classifier to predict delayed flights, 
using cross validation to choose the best values for model parameters.

You'll find good values for the following parameters:

featureSubsetStrategy — the number of features to consider for splitting at each node and
maxDepth — the maximum number of splits along any branch.
Unfortunately building this model takes too long, 
so we won't be running the .fit() method on the pipeline.
"""
Subset of data from the flights DataFrame:

+---+------+--------+-----------------+-----+
|mon|depart|duration|features         |label|
+---+------+--------+-----------------+-----+
|9  |10.33 |195     |[9.0,10.33,195.0]|0    |
|1  |8.0   |232     |[1.0,8.0,232.0]  |0    |
|11 |7.77  |60      |[11.0,7.77,60.0] |1    |
|4  |13.25 |210     |[4.0,13.25,210.0]|0    |
|3  |17.58 |265     |[3.0,17.58,265.0]|1    |
+---+------+--------+-----------------+-----+
only showing top 5 rows

# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder() \
            .addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2']) \
            .addGrid(forest.maxDepth, [2, 5, 10]) \
            .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(estimator=forest, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
"""
Evaluating Random Forest
In this final exercise you'll be evaluating the results of cross-validation on a Random Forest model.

The following have already been created:

cv - a cross-validator which has already been fit to the training data
evaluator — a BinaryClassificationEvaluator object and
flights_test — the testing data.
"""

# Average AUC for each parameter combination in grid
avg_auc = cv.avgMetrics
print(avg_auc)
"""
[0.61550451929848, 0.661275302749083, 0.6832959983649716, 0.6790399103856084, 
0.6404890400309002, 0.6659871420567183, 0.6808977119243277, 0.6867946590518151, 
0.6414270561540629, 0.6653385916148042, 0.6832494433718275, 0.6851695159338953, 
0.6414270561540629, 0.6653385916148042, 0.6832494433718275, 0.6851695159338953]
"""
# Average AUC for the best model
best_model_auc = max(cv.avgMetrics)
print(best_model_auc)
"""0.6867946590518151"""

# What's the optimal parameter value?
opt_max_depth = cv.bestModel.explainParam('maxDepth')
opt_feat_substrat = cv.bestModel.explainParam('featureSubsetStrategy')
print(opt_max_depth)
"""
maxDepth: Maximum depth of the tree. (>= 0) E.g., 
depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (current: 20)
"""
print(opt_feat_substrat)
"""
featureSubsetStrategy: The number of features to consider for splits at each tree node. 
Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n]. (current: onethird)
"""

# AUC for best model on testing data
best_auc = evaluator.evaluate(cv.transform(flights_test))
print(best_auc)
"""
0.6966021421117832"""