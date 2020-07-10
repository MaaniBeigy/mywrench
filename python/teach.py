# Statistical Programming in Python
from matplotlib import pyplot as plt  # Visualization with Python
import seaborn as sns  # Statistical data visualization
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Python Data Analysis Library
import os  # Miscellaneous operating system interfaces
from pathlib import Path  # Object-oriented filesystem paths
print("Hello World")
# -------------------------- Installing and Preparation -----------------------
# Check Python Version -> In eTerminal
## python --version

# If pip isn’t already installed, try to bootstrap it from the standard library:
## pip --version
# python -m ensurepip --default-pip

# Use pip for Installing -> In Terminal
# pip install seaborn
# -------------------------------- Introduction -------------------------------
# load required libraries
# find helps or documentations
help(sns.boxplot)
df = sns.load_dataset('iris')  # load iris dataset as dataframe
type(df)  # check the type of the dataframe
# ---------------------------- Draw and Show a Plot ---------------------------
# Make boxplot for one group only
plt.figure(figsize=(8, 8), dpi=100)  # figsize: width, height in inches
fig1 = sns.boxplot(y=df["sepal_length"])   # make the fig1 as boxplot
plt.show()
# ---------------------------- Draw and Save a Plot ---------------------------
plt.figure(figsize=(8, 8), dpi=100)
fig1 = sns.boxplot(y=df["sepal_length"])
# plt.savefig(
#     os.path.join(  # absolute path
#         'address goes here',
#         'fig1.pdf'
#         ),
#     format = 'pdf'
#     )
# -------------------------------- Relative Paths -----------------------------
cwd = Path.cwd()  # find the current working directory (cwd)
print(cwd)
data_path = (cwd / './data/').resolve()  # determine data path
print(data_path)
figures_path = (cwd / './figures/').resolve()  # determine figures path
print(figures_path)
# Draw and Save a Plot using Relative Path
plt.figure(figsize=(8, 8), dpi=100)
fig1 = sns.boxplot(y=df["sepal_length"])
plt.savefig(os.path.join(figures_path, 'fig1.pdf'), format='pdf')
# --------------------------- Assignment and Operations ----------------------
weight = 72.0  # weight in kilograms
height = 1.82  # height in meters
BMI = weight/(height**2)  # formula to calculate Body Mass Index (kg/m^2)
BMI

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
dict_var = {1: "a", 2: "b", 3: "c", 4: "d"}
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

len(log_array)
# --------------------------------- DataFrames --------------------------------
df = pd.DataFrame({
    'id': list(range(1, 4)),
    'gender': pd.Categorical(["m", "f", "m"]),
    'fbs': np.array([104, 98, 129])
})
df.shape  # dimensions of a dataframe
df.dtypes  # data types of variables in a dataframe
df.head()  # view just a head of dataframe
df.describe()  # describe dataframe
df.columns  # shows the column names of datafram
df.T  # transpose
type(df)  # shows the type of a dataframe
# check if an object is of class pandas dataframe
isinstance(df, pd.core.frame.DataFrame)
# ------------------------ Selection/Subsetting DataFrames --------------------
df.columns[0]  # shows the name of the first column of a dataframe
df['id']  # selecting a single column
df[:1]  # selecting a single row
# select via the label
df.loc[:, ['gender', 'fbs']]  # selecting on a multi-axis by label
df.loc['0':'1', ['id', 'gender']]  # showing label slicing
df.at[0, 'fbs']
# select via the position of the passed integers
df.iloc[2]  # select data of row with index 2, i.e., 3rd row
df.iloc[0:2, 0:2]  # select by integer slices
df.iloc[1:3, :]  # slicing rows explicitly
df.iloc[:, 0:2]  # slicing columns explicitly
df.iloc[1, 1]  # getting a value explicitly
# select via Boolean indexing i.e. conditions
df[df['fbs'] > 100]  # using subsetting approach 
df.query('fbs > 100')  # using query approach
df2 = df.copy()  # make a copy of dataframe for further handlings
df2.loc[3] = [4, 'other', np.nan]  # add another row to dataframe
df2[df2['gender'].isin(['m', 'f'])]  # using the isin() method for filtering
pd.isna(df2)  # To get the boolean mask where values are nan (i.e., missing)
df2.dropna(how='any')  # To drop any rows that have missing data.
# ---------------------------- Operations on DataFrames ------------------------
df['fbs'].mean()  # performing a descriptive statistic:
df['fbs'].median()  # median of a variable
df['fbs'].std()  # standard deviation of a variable
df['fbs'].min()  # minimum of a variable
df['fbs'].sum()  # sum of a variable

# applying functions to the data
df.apply(np.cumsum)
# concat dataframes
df3 = pd.DataFrame(np.random.randn(10, 4))
pieces = [df3[:3], df3[3:7], df3[7:]]
df3_concat = pd.concat(pieces)
df3_concat == df3
# join data frames by SQL style merges
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
# grouping variables
df4 = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})
df4.head()
df4.groupby('A').sum()
df4.groupby(['A', 'B']).mean()
df4.groupby(['A', 'B']).size()
df4.groupby('A').agg({'D': 'mean'}) # aggregate method
# mutating/assigning new variables
df4 = df4.assign(E = df4['D'] - df4['C'])
# renaming column/variable names 
df4 = df4.rename(columns = {'E': 'Subtract'})
# pivot tables
pd.pivot_table(df4, values='D', index=['A'], columns=['B'])
# Sorting dataframe
df4.sort_values(by="C")
# ----------------------------- Reading/Writing Data ---------------------------
# reading from csv
df5 = pd.read_csv(
    os.path.join(data_path, 'df.csv')
)
df5.shape  # dimensions of a dataframe
df5.dtypes  # data types of variables in a dataframe
df5.head()  # view just a head of dataframe
df5.describe()  # describe dataframe
df5['gender'].nunique()  # get the unique number of categries
# writing to csv
df5.to_csv(
    os.path.join(data_path, 'df.csv')
)
# writing to a HDF5 store (Apache Hadoop Distributed File System)
df5.to_hdf(
    path_or_buf=os.path.join(data_path, 'df5.h5'),
    key='df5', 
    mode='w'
    )
# reading from HDF5
df6 = pd.read_hdf(
    os.path.join(data_path, 'df5.h5')
    )
all(df6 == df5)
# reading from excel
df7 = pd.read_excel(
    os.path.join(data_path, 'df.xlsx')
)
all(df7 == df6)
