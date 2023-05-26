# Databricks notebook source
# DBTITLE 1,Loading the data
import pandas as pa
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from pyspark.ml. feature import VectorAssembler 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df = spark.read. format ("csv") .option("header", True) .option ("inferschema", True) . load ("/FileStore/tables/Motor_Vehicle_Collisions___Crashes.csv")
display (df)

# COMMAND ----------

# DBTITLE 1,Pre Processing Steps
columns_to_drop = ['CRASH TIME', 'ON STREET NAME', 'LATITUDE', 'LONGITUDE', 'LOCATION', 'ON STREET NAME',  'CROSS STREET NAME', 'OFF STREET NAME', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']
df = df.drop(*columns_to_drop)
display (df)

# COMMAND ----------

df = df.withColumnRenamed("CONTRIBUTING FACTOR VEHICLE 1","CAUSES")
display(df)

# COMMAND ----------

# DBTITLE 1,Eliminating Missing Rows
# Drop rows with missing values
df = df.dropna()
display(df)

# COMMAND ----------

print(df.count())

# COMMAND ----------

from pyspark.sql.functions import col

df = df.withColumn("ZIP CODE", col("ZIP CODE").cast("int"))
df = df.withColumn("NUMBER OF CYCLIST KILLED", col("NUMBER OF CYCLIST KILLED").cast("int"))
df = df.withColumn("COLLISION_ID", col("COLLISION_ID").cast("int"))

# COMMAND ----------

df = df.drop_duplicates()

# COMMAND ----------

# Save the cleaned data
df.write.csv('bicycle_crash_report.csv', header=True, mode='overwrite')

# COMMAND ----------

from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# Map label values to binary labels
df = df.withColumn("label", when((col("NUMBER OF CYCLIST INJURED") + col("NUMBER OF CYCLIST KILLED")) > 0, 1).otherwise(0))

# Create feature columns and assemble them into a feature vector
feature_columns = ["BOROUGH", "CAUSES"]
tokenizer = RegexTokenizer(inputCol="CAUSES", outputCol="tokens", pattern="\\W")
stopwords = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_tokens")
cv = CountVectorizer(inputCol=stopwords.getOutputCol(), outputCol="raw_features", vocabSize=1000)
idf = IDF(inputCol=cv.getOutputCol(), outputCol="features", minDocFreq=5)
assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
indexer = StringIndexer(inputCol="BOROUGH", outputCol="borough_index")

# Train a logistic regression model to predict the main causes of accidents involving cyclists
lr = LogisticRegression(featuresCol="features_vector", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, stopwords, cv, idf, assembler, indexer, lr])

# Split the data into training and testing sets
trainingData1, testData1 = df.randomSplit([0.8, 0.2], seed=42)

# Train the pipeline on the training data
model1 = pipeline.fit(trainingData1)

# Make predictions on the testing data
predictions1 = model1.transform(testData1)

# Evaluate the model
correct1 = predictions1.filter(col('prediction') == col('label')).count()
total1 = predictions1.count()
accuracy1 = correct1/total1
print('Accuracy:', accuracy1)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA

# Select relevant columns
tdf = df.select('CAUSES')

# Tokenize the text column
tokenizer = RegexTokenizer(inputCol='CAUSES', outputCol='words', pattern='\\W')
tdf = tokenizer.transform(tdf)

# Remove stop words
stopwords = StopWordsRemover(inputCol='words', outputCol='filtered')
tdf = stopwords.transform(tdf)

# Convert the text to a vector of term frequency
cv = CountVectorizer(inputCol='filtered', outputCol='rawFeatures')
cvmodel = cv.fit(tdf)
tdf = cvmodel.transform(tdf)

# Compute IDF and transform the feature vectors
idf = IDF(inputCol='rawFeatures', outputCol='features')
idfModel = idf.fit(tdf)
tdf = idfModel.transform(tdf)

# Train an LDA model to identify topics
lda = LDA(k=10, maxIter=10)
ldaModel = lda.fit(tdf)

# Print the topics and their corresponding top terms
topics = ldaModel.describeTopics(5)
for topic in topics.rdd.collect():
    print('Topic: ' + str(topic.topic))
    for i in topic.termIndices:
        print('   Term ' + str(i+1) + ': ' + cvmodel.vocabulary[i])
        if i == topic.termIndices[0]:
            print('   Top Term')
