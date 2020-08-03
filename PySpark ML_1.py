# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:51:38 2020

@author: RikSo
"""

"""
Creating a SparkSession
In this exercise, you'll spin up a local Spark cluster using all available cores. The cluster will be accessible via a SparkSession object.

The SparkSession class has a builder attribute, which is an instance of the Builder class. The Builder class exposes three important methods that let you:

specify the location of the master node;
name the application (optional); and
retrieve an existing SparkSession or, if there is none, create a new one.
The SparkSession class has a version attribute which gives the version of Spark.

Find out more about SparkSession here.

Once you are finished with the cluster, it's a good idea to shut it down, which will free up its resources, making them available for other processes.
"""
# Import the PySpark module
from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()


"""
Loading flights data
In this exercise you're going to load some airline flight data from a CSV file. To ensure that the exercise runs quickly these data have been trimmed down to only 50 000 records. You can get a larger dataset in the same format here.

Notes on CSV format:

fields are separated by a comma (this is the default separator) and
missing data are denoted by the string 'NA'.
Data dictionary:

mon — month (integer between 1 and 12)
dom — day of month (integer between 1 and 31)
dow — day of week (integer; 1 = Monday and 7 = Sunday)
org — origin airport (IATA code)
mile — distance (miles)
carrier — carrier (IATA code)
depart — departure time (decimal hour)
duration — expected duration (minutes)
delay — delay (minutes)
pyspark has been imported for you and the session has been initialized.

Note: The data have been aggressively down-sampled.
"""

# Read data from CSV file
flights = spark.read.csv('flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())
"""
The data contain 50000 records.
"""

# View the first five records
flights.show(5)
"""
+---+---+---+-------+------+---+----+------+--------+-----+
|mon|dom|dow|carrier|flight|org|mile|depart|duration|delay|
+---+---+---+-------+------+---+----+------+--------+-----+
| 11| 20|  6|     US|    19|JFK|2153|  9.48|     351| null|
|  0| 22|  2|     UA|  1107|ORD| 316| 16.33|      82|   30|
|  2| 20|  4|     UA|   226|SFO| 337|  6.17|      82|   -8|
|  9| 13|  1|     AA|   419|ORD|1236| 10.33|     195|   -5|
|  4|  2|  5|     AA|   325|ORD| 258|  8.92|      65| null|
+---+---+---+-------+------+---+----+------+--------+-----+
only showing top 5 rows
"""
# Check column data types
flights.dtypes
"""
[('mon', 'int'),
 ('dom', 'int'),
 ('dow', 'int'),
 ('carrier', 'string'),
 ('flight', 'int'),
 ('org', 'string'),
 ('mile', 'int'),
 ('depart', 'double'),
 ('duration', 'int'),
 ('delay', 'int')]
"""
"""
Read data from a CSV file called 'flights.csv'. Assign data types to columns automatically. Deal with missing data.
How many records are in the data?
Take a look at the first five records.
What data types have been assigned to the columns? Do these look correct?
"""



"""
Loading SMS spam data
You've seen that it's possible to infer data types directly from the data. 
Sometimes it's convenient to have direct control over the column types. 
You do this by defining an explicit schema.
The file sms.csv contains a selection of SMS messages 
which have been classified as either 'spam' or 'ham'. 
These data have been adapted from the UCI Machine Learning Repository. 
There are a total of 5574 SMS, of which 747 have been labelled as spam.

Notes on CSV format:

no header record and
fields are separated by a semicolon (this is not the default separator).
Data dictionary:

id — record identifier
text — content of SMS message
label — spam or ham (integer; 0 = ham and 1 = spam)


First few records from 'sms.csv':

1;Sorry, I'll call later in meeting;0
2;Dont worry. I guess he's busy.;0
3;Call FREEPHONE 0800 542 0578 now!;1
4;Win a 1000 cash prize or a prize worth 5000;1

"""

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()






