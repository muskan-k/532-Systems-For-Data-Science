# Importing necessary libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession 
from pyspark.streaming import StreamingContext 
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row 
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
import json
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
import pandas as pd
import pickle
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
import logging

# Setting up Spark Context and Streaming Context

# Initializing SparkContext with local mode and 2 threads
sc = SparkContext('local[2]','stream_test') 
sc.setLogLevel("WARN") # Setting logging level for SparkContext
# Initializing StreamingContext with batch interval of 15 seconds
ssc = StreamingContext(sc, 15)
spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate() # Initializing SparkSession
spark.sparkContext.setLogLevel("ERROR") # Setting logging level for SparkSession

# Creating a DStream to read data from socket at port 6100
lines = ssc.socketTextStream("localhost", 6100)

# Defining schema for the input data with schema fields
Schema = StructType([
    StructField('number_of_elements', FloatType(), True),
    StructField('mean_atomic_mass', FloatType(), True),
    StructField('wtd_mean_atomic_mass', FloatType(), True),
    StructField('gmean_atomic_mass', FloatType(), True),
    StructField('wtd_gmean_atomic_mass', FloatType(), True),
    StructField('entropy_atomic_mass', FloatType(), True),
    StructField('wtd_entropy_atomic_mass', FloatType(), True),
    StructField('mean_atomic_radius', FloatType(), True),
    StructField('wtd_mean_atomic_radius', FloatType(), True),
    StructField('gmean_atomic_radius', FloatType(), True),
    StructField('wtd_gmean_atomic_radius', FloatType(), True),
    StructField('entropy_atomic_radius', FloatType(), True),
    StructField('wtd_entropy_atomic_radius', FloatType(), True),
    StructField('range_atomic_radius', FloatType(), True),
    StructField('wtd_range_atomic_radius', FloatType(), True),
    StructField('mean_Density', FloatType(), True),
    StructField('wtd_mean_Density', FloatType(), True),
    StructField('gmean_Density', FloatType(), True),
    StructField('wtd_gmean_Density', FloatType(), True),
    StructField('mean_ElectronAffinity', FloatType(), True),
    StructField('wtd_mean_ElectronAffinity', FloatType(), True),
    StructField('mean_ThermalConductivity', FloatType(), True),
    StructField('wtd_mean_ThermalConductivity', FloatType(), True),
    StructField('wtd_entropy_ThermalConductivity', FloatType(), True),
    StructField('range_ThermalConductivity', FloatType(), True),
    StructField('wtd_range_ThermalConductivity', FloatType(), True),
    StructField('std_ThermalConductivity', FloatType(), True),
    StructField('wtd_std_ThermalConductivity', FloatType(), True),
    StructField('mean_Valence', FloatType(), True),
    StructField('wtd_mean_Valence', FloatType(), True),
    StructField('gmean_Valence', FloatType(), True),
    StructField('wtd_gmean_Valence', FloatType(), True),
    StructField('range_Valence', FloatType(), True),
    StructField('wtd_range_Valence', FloatType(), True),
    StructField('critical_temp', FloatType(), True)
])

# Function to preprocess the data
def preproc(words):
    # Collecting data from each RDD
	rdds=words.collect()
	values=list()
	for j in rdds:
        # Extracting values from JSON strings
		values=[i for i in list(json.loads(j).values())] 
	if len(values)==0:
		return 
    # Creating DataFrame from collected data
	df=spark.createDataFrame((Row(**d) for d in values),schema=Schema)
	count = df.count() # Counting number of records in the DataFrame
    # Calling process function to train models
	process(df, count)
	
# Function to process data and train models
def process(df, count):
    # Assembling 34 features into a single vector in a new column named "features"
    assembler = VectorAssembler(inputCols=df.columns[0:-1], outputCol="features")
    data_set = assembler.transform(df)
    # Selecting features and target column
    data_set = data_set.select(["features", "critical_temp"])
    # Splitting data into training and testing sets
    (train, test) = data_set.randomSplit([0.8,0.2], seed=3500)
    
    # Training Random Forest Regressor
    rfRegressor = RandomForestRegressor(featuresCol="features",labelCol='critical_temp')
    rfModel = rfRegressor.fit(train)
    # Saving Random Forest Regressor model
    rfFile="RandomForestRegressorModel.pkl"
    pickle.dump(rfFile, open(rfFile,"wb"))
    # Making predictions and evaluating
    test_stats = rfModel.transform(test)
    predictions = test_stats.select('critical_temp','prediction')
    # Writing the predictions and actual values dataframe into a csv file
    write_results_to_file("RandomForestPredictions.csv", predictions)
    # Evaluating for RMSE metric
    evaluator = RegressionEvaluator(labelCol="critical_temp", predictionCol = "prediction", metricName = "rmse")
    rmse = evaluator.evaluate(test_stats)
    # Writing batch size and RMSE value to a new file
    write_to_file("RandomForestRegressorResults.txt", count, rmse)

    # Training Decision Tree Regressor
    drRegressor = DecisionTreeRegressor(featuresCol="features",labelCol='critical_temp')
    drModel = drRegressor.fit(train)
    # Saving Decision Tree model
    drFile="DecisionTreeRegressorModel.pkl"
    pickle.dump(drFile, open(drFile,"wb"))
    # Making predictions and evaluating
    test_stats = drModel.transform(test)
    predictions = test_stats.select('critical_temp','prediction')
    write_results_to_file("DecisionTreePredictions.csv", predictions)
    evaluator = RegressionEvaluator(labelCol="critical_temp", predictionCol = "prediction", metricName = "rmse")
    rmse = evaluator.evaluate(test_stats)
    write_to_file("DecisionTreeRegressorResults.txt", count, rmse)

    # Training Gradient Boosted Tree Regressor
    gbtRegressor = GBTRegressor(featuresCol="features",labelCol='critical_temp')
    gbtModel = gbtRegressor.fit(train)
    # Saving Gradient Boosted Tree model
    gbtFile="GradientBoostedTreeRegressorModel.pkl"
    pickle.dump(gbtFile, open(gbtFile,"wb"))
    # Making predictions and evaluating
    test_stats = gbtModel.transform(test)
    predictions = test_stats.select('critical_temp','prediction')
    write_results_to_file("GradientBoostedTreePredictions.csv", predictions)
    evaluator = RegressionEvaluator(labelCol="critical_temp", predictionCol = "prediction", metricName = "rmse")
    rmse = evaluator.evaluate(test_stats)
    write_to_file("GradientBoostedTreeRegressorResults.txt", count, rmse)

    # Plotting RMSE vs Batch Size graph for each model
    cwd = os.getcwd()
    rmse_path = os.path.join(cwd, "RMSEResults") # Getting PWD and adding RMSE results folder to the path 
    plot(os.path.join(rmse_path, "RandomForestRegressorResults.txt"), "RandomForestRegressor")
    plot(os.path.join(rmse_path, "DecisionTreeRegressorResults.txt"), "DecisionTreeRegressor")
    plot(os.path.join(rmse_path, "GradientBoostedTreeRegressorResults.txt"), "GradientBoostedTreeRegressor")

# Function to write evaluation results to file
def write_to_file(filename, batch_size, rmse):
    cwd = os.getcwd()
    # Setting output directory name
    output_directory = "RMSEResults"
    # Creating output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    with open(os.path.join(output_directory, filename), 'a') as file:
        file.write(f"{batch_size} {rmse}\n")
    return

# Function to write predictions to a csv file 
def write_results_to_file(filename, predictions):
    cwd = os.getcwd()
    predictions = predictions.toPandas()
    output_directory = "Predictions" # Setting output directory name
    os.makedirs(output_directory, exist_ok=True) # Creating output directory if it doesn't exist
    predictions.to_csv(os.path.join(output_directory, filename), index=False)

# Function to plot RMSE vs Batch Size graph
def plot(filename, model_name):
    batch_sizes = []
    rmse_scores = []
    
    # Create the "Graphs" folder if it doesn't exist
    output_directory = "Graphs"
    os.makedirs(output_directory, exist_ok=True)
    
    # Open the RMSE results file and read its contents
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split(' ')
            batch_size = int(parts[0])
            rmse = float(parts[1])
            batch_sizes.append(batch_size)
            rmse_scores.append(rmse)

    # Plot the graph
    plt.plot(batch_sizes, rmse_scores, label=model_name)
    plt.xlabel('Batch Size')
    plt.ylabel('RMSE')
    plt.title('Batch Size vs RMSE')
    plt.legend()
    plt.grid(True)
    
    # Save the plot in the "Graphs" folder
    plt.savefig(os.path.join(output_directory, f'{model_name}_graph.png'))

# Applying preproc function to each RDD in the DStream 
lines.foreachRDD(lambda rdd: preproc(rdd))

# Start the Streaming Context
ssc.start()

# Print active context
print(ssc.getActive())

# Wait for the computation to terminate
ssc.awaitTermination()

# Stop the Streaming Context
ssc.stop(True,True)

# Close the Streaming Context
ssc.close()

