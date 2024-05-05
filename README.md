# Prediction of Critical Temperature of Superconductors using SparkML

CS532 - Systems for Data Science

Team Members:
1. Muskan Kothari
2. Ujjwal Gupta

## Superconductivity Dataset

Download and extract the dataset from [here](https://archive.ics.uci.edu/dataset/464/superconductivty+data). We have reduced the dataset features to only 34 out of the original 81 contained in this dataset.

## Requirements

Install python3.12 by executing the following commands :

```bash
$ sudo apt update
$ sudo apt install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt install python3.12
```
Create a virtual environment before installing all the necessary requirements 
```
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv venv
source venv/bin/activate
```

Use `Python3.12` to install the libraries defined in the requirements.txt file 
while the environment is activated in the projects directory

```
pip3 install -r requirements.txt
```

In case of ModuleNotFoundError related to distutils:
```
pip install setuptools
```

## Build

```bash
$ sudo apt update && sudo apt upgrade
$ sudo apt install openjdk-8-jdk
$ wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
```
Extract the tar file downloaded for Spark. 

Finally, in order to set the logging properties for Spark, execute the following code:

```bash
$ cd /opt/spark
$ cp conf/log4j.properties.template conf/log4j.properties
```

Open `conf/log4j.properties` file and change the following line

```bash
log4j.rootCategory=INFO, console
```

to

```bash
log4j.rootCategory=WARN, console
```

After making the above changes execute ```source ~/.bashrc && source ~/.profile```

## Streaming Dataset

Execute the following code in one terminal under the project directory to start streaming the dataset.

```bash
$ python3 ./stream.py --file="chemical_features" --batch-size=5000
```

## Executing the spark driver code in another terminal

Execute the following command in second terminal to execute the driver code.

```bash
$ ~/spark-3.5.1-bin-hadoop3/bin/spark-submit analysis.py
```
