# Statistical Programming in Python
print("Hello World")
# -------------------------- Installing and Preparation -----------------------
# Check Python Version -> In eTerminal
## python --version

# If pip isn’t already installed, try to bootstrap it from the standard library:
## pip --version
## python -m ensurepip --default-pip

# Use pip for Installing -> In Terminal
## pip install seaborn
# -------------------------------- Introduction -------------------------------
# load required libraries
from matplotlib import pyplot as plt  # Visualization with Python
import numpy as np  # Fundamental package for scientific computing with Python
import seaborn as sns  # Statistical data visualization
import pandas as pd  # Python Data Analysis Library
# find helps or documentations
help(sns.boxplot)
df = sns.load_dataset('iris')  # load iris dataset as dataframe
type(df)  # check the type of the dataframe
# ---------------------------- Draw and Show a Plot ---------------------------
# Make boxplot for one group only
plt.figure(figsize = (8, 8), dpi = 100)  # figsize: width, height in inches
fig1 = sns.boxplot(y = df["sepal_length"])   # make the fig1 as boxplot
plt.show()
# ---------------------------- Draw and Save a Plot ---------------------------
import os  # Miscellaneous operating system interfaces
plt.figure(figsize = (8, 8), dpi = 100)
fig1 = sns.boxplot(y = df["sepal_length"])
# plt.savefig(
#     os.path.join(  # absolute path
#         'address goes here',
#         'fig1.pdf'
#         ),
#     format = 'pdf'
#     )
# -------------------------------- Relative Paths -----------------------------
from pathlib import Path  # Object-oriented filesystem paths
import os
cwd = Path.cwd()  # find the current working directory (cwd)
print(cwd)
data_path = (cwd / './data/').resolve()  # determine data path
print(data_path)
figures_path = (cwd / './figures/').resolve()  # determine figures path
print(figures_path)
# Draw and Save a Plot using Relative Path
plt.figure(figsize = (8, 8), dpi = 100)
fig1 = sns.boxplot(y = df["sepal_length"])
plt.savefig(os.path.join(figures_path, 'fig1.pdf'),format = 'pdf')
# --------------------------- Assignment and Operations ----------------------
weight = 72.0  # weight in kilograms
height = 1.82  # height in meters
BMI = weight/(height**2)  # formula to calculate Body Mass Index (kg/m^2)
BMI

import numpy as np

values1 = list(range(1, 101))  # create a list
print(values1)  # print the list values
np.mean(values1)  # find the mean of the list
# Create an array with missing values (i.e., nan)
values2 = np.array([1, np.nan, 3, 4])  
print(values2)  # print the list values
np.mean(values2)  # find the mean of the array
np.nanmean(values2)  # find the mean of the array and remove nan

# ---------------------------- Functions and Conditions -----------------------
def BodyMassIndex(weight, height): 
    if height > 2.5:
        raise ValueError('height is not in meters')
    else:
        return(weight/(height**2)) 
BodyMassIndex(72, 1.82)
# ----------------------------- Data Types and Classes ------------------------
# -------------------------------- scalar variables ---------------------------
# a float variable is a numeric with fractional partts
float_var = 8.4 
type(float_var)
isinstance(float_var, float)
# an ineteger is Positive or negative whole number (without a fractional part) 
int_var = 6
type(int_var)
isinstance(int_var, int)
# a string variable is a collection of one or more characters
str_var = "foo"  
type(str_var)
isinstance(str_var, str)
# a boolean variable is composed of True/False data
bool_var = True  
type(bool_var)
isinstance(bool_var, bool)
# ---------------------------- Dictionaries and Lists -------------------------
# a dictionary is an unordered collection key:value pair of data
dict_var = {1:"a", 2:"b", 3:"c", 4: "d"}  
type(dict_var)
isinstance(dict_var, dict)
# a list is an ordered collection of data, not necessarily of the same type
list_var = [1, 2, 3, 4] 
type(list_var)
isinstance(list_var, list)

# a tuple is an ordered collection of data, not necessarily of the same type
tuple_var = (1, 2, 3, 4)
type(tuple_var)
isinstance(tuple_var, tuple)
# -------------------------------- Numpy Arrays -------------------------------
import numpy as np

float_array = np.array([1, 2.5, 4.5])
float_array.dtype.name
isinstance(float_array[0], float)
isinstance(float_array[0], np.float_)
isinstance(float_array[0], np.double)


str_array = np.array(['1', 'a', 4.5])  # notice type coersion
str_array.dtype.name
isinstance(str_array[0], np.str_)

int_array = np.array([1, 6, 10])
int_array.dtype.name
isinstance(int_array[0], np.int_)

log_array = np.array([True, False, True, False])
log_array.dtype.name
isinstance(log_array[0], np.bool)
isinstance(log_array[0], np.bool8)
isinstance(log_array[0], np.bool_)

