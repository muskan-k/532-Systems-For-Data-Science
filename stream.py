#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Run using python3 stream.py -f <input_file> -b <batch_size> to use a custom file/dataset and batch size
# Run using python3 stream.py -e True to stream endlessly in a loop
parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context') # Creating argument parser
parser.add_argument('--file', '-f', help='File to stream', required=False,
                    type=str, default="chemical_features")     # Argument for input file
parser.add_argument('--batch-size', '-b', help='Batch size',
                    required=False, type=int, default=100)  # Argument for batch size, default being 100
parser.add_argument('--endless', '-e', help='Enable endless stream',
                    required=False, type=bool, default=False)  # Argument for endless streaming

TCP_IP = "localhost"
TCP_PORT = 6100

# Function to connect to TCP server
def connectTCP():  
    # Creating a TCP/IP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Setting socket option to allow reusing the address
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Binding socket to IP and port
    s.bind((TCP_IP, TCP_PORT))
    # Listening for connections
    s.listen(10)
    print(f"Waiting for connection on port {TCP_PORT}...")
    connection, address = s.accept() # Accepting connection
    print(f"Connected to {address}") 
    return connection, address

# Function to stream a dataset
def streamDataset(tcp_connection, dataset_type):  
    print(f"Starting to stream {dataset_type} dataset")
    # List of files in dataset to stream
    DATASETS = [    
        "train"
    ]
    for dataset in DATASETS:
        # Streaming CSV file
        streamCSVFile(tcp_connection, f'{dataset_type}/{dataset}.csv')
        # Sleep to simulate real-time streaming
        time.sleep(5)

# Function to stream a CSV file to Spark
def streamCSVFile(tcp_connection, input_file): 
    '''
    Each batch is streamed as a JSON file and has the following shape. 
    The outer indices are the indices of each row in a batch and go from 0 - batch_size-1
    The inner indices are the indices of each column in a row and go from 0 - feature_size-1

    {
        '0':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
        '1':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
        ...
        'batch_size-1':{
            'feature0': <value>,
            'feature1': <value>,
            ...
            'featureN': <value>,
        }
    }
    '''

    df = pd.read_csv(input_file)  # Reading CSV file into DataFrame
    values = df.values.tolist()  # Converting DataFrame to list

    # Looping through batches of size batch_size lines
    for i in tqdm(range(0, len(values)-batch_size+2, batch_size)):
        send_data = values[i:i+batch_size]  # Loading batch of rows
        payload = dict()    # Creating payload
        # Iterate over the batch
        for mini_batch_index in range(len(send_data)):
            payload[mini_batch_index] = dict()  # Create a record
            # Iterate over the features
            for feature_index in range(len(send_data[0])):
                # Add the feature to the record
                payload[mini_batch_index][f'feature{feature_index}'] = send_data[mini_batch_index][feature_index]
        # Encoding payload and adding newline character
        send_batch = (json.dumps(payload) + '\n').encode()
        try:
            tcp_connection.send(send_batch)  # Send the payload to Spark
        except BrokenPipeError:  # Handling broken pipe error
            print("Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        # Sleep to simulate real-time streaming
        time.sleep(5)

connection_delay = 60 
if __name__ == '__main__':
    args = parser.parse_args()  # Parsing command line arguments
    print(args)  # Printing parsed arguments

    input_file = args.file  # Getting input file from arguments
    batch_size = args.batch_size  # Getting batch size from arguments
    endless = args.endless  # Getting endless streaming flag from arguments

    tcp_connection, _ = connectTCP()  # Connecting to TCP server
    _function = streamDataset   # Assigning function based on dataset
   
    if endless:
        while True:
            _function(tcp_connection, input_file)  # Streaming dataset endlessly
    else:
        _function(tcp_connection, input_file)  # Streaming dataset once
    time.sleep(connection_delay)
    tcp_connection.close()  # Closing TCP connection
