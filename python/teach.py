"""Statistical Programming in Python."""

import os  # Miscellaneous operating system interfaces
from pathlib import Path  # Object-oriented filesystem paths

import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Python Data Analysis Library
import seaborn as sns  # Statistical data visualization
from matplotlib import pyplot as plt  # Visualization with Python
from sqlalchemy import create_engine  # sqlalchemy engine

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
MYWEIGHT = 72.0  # weight in kilograms
MYHEIGHT = 1.82  # height in meters
BMI = MYWEIGHT/(MYHEIGHT**2)  # formula to calculate Body Mass Index (kg/m^2)
print(BMI)

values1 = list(range(1, 101))  # create a list
print(values1)  # print the list values
np.mean(values1)  # find the mean of the list
# Create an array with missing values (i.e., nan)
values2 = np.array([1, np.nan, 3, 4])
print(values2)  # print the list values
np.mean(values2)  # find the mean of the array
np.nanmean(values2)  # find the mean of the array and remove nan

# ---------------------------- Functions and Conditions -----------------------


def body_mass_index(weight, height):
    """Body mass index."""
    if height > 2.5:
        raise ValueError('height is not in meters')
    else:
        return weight/(height**2)


body_mass_index(72, 1.82)
# ----------------------------- Data Types and Classes ------------------------
# -------------------------------- scalar variables ---------------------------
# a float variable is a numeric with fractional parts
FLOAT_VAR = 8.4
type(FLOAT_VAR)
isinstance(FLOAT_VAR, float)
# an integer is Positive or negative whole number (without a fractional part)
INT_VAR = 6
type(INT_VAR)
isinstance(INT_VAR, int)
# a string variable is a collection of one or more characters
STR_VAR = "foo"
type(STR_VAR)
isinstance(STR_VAR, str)
# a boolean variable is composed of True/False data
BOOL_VAR = True
type(BOOL_VAR)
isinstance(BOOL_VAR, bool)
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
print(float_array.dtype.name)
isinstance(float_array[0], float)
isinstance(float_array[0], np.float_)
isinstance(float_array[0], np.double)

str_array = np.array(['1', 'a', 4.5])  # notice type coercion
print(str_array.dtype.name)
isinstance(str_array[0], np.str_)

int_array = np.array([1, 6, 10])
print(int_array.dtype.name)
isinstance(int_array[0], np.int_)

log_array = np.array([True, False, True, False])
print(log_array.dtype.name)
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
print(df.shape)  # dimensions of a dataframe
print(df.dtypes)  # data types of variables in a dataframe
print(df.head())  # view just a head of dataframe
print(df.describe())  # describe dataframe
print(df.columns)  # shows the column names of dataframe
print(df.T)  # transpose
type(df)  # shows the type of a dataframe
# check if an object is of class pandas dataframe
isinstance(df, pd.core.frame.DataFrame)
# ------------------------ Selection/Subsetting DataFrames --------------------
print(df.columns[0])  # shows the name of the first column of a dataframe
print(df['id'])  # selecting a single column
print(df[:1])  # selecting a single row
# select via the label
print(df.loc[:, ['gender', 'fbs']])  # selecting on a multi-axis by label
print(df.loc['0':'1', ['id', 'gender']])  # showing label slicing
print(df.at[0, 'fbs'])
# select via the position of the passed integers
print(df.iloc[2])  # select data of row with index 2, i.e., 3rd row
print(df.iloc[0:2, 0:2])  # select by integer slices
print(df.iloc[1:3, :])  # slicing rows explicitly
print(df.iloc[:, 0:2])  # slicing columns explicitly
print(df.iloc[1, 1])  # getting a value explicitly
# select via Boolean indexing i.e. conditions
print(df[df['fbs'] > 100])  # using subsetting approach
df.query('fbs > 100')  # using query approach
df2 = df.copy()  # make a copy of dataframe for further handlings
df2.loc[3] = [4, 'other', np.nan]  # add another row to dataframe
print(df2[df2['gender'].isin(['m', 'f'])])  # using the isin() method for filtering
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
print(df3_concat == df3)
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
print(df5.shape)  # dimensions of a dataframe
print(df5.dtypes)  # data types of variables in a dataframe
print(df5.head())  # view just a head of dataframe
print(df5.describe())  # describe dataframe
df5['gender'].nunique()  # get the unique number of categories
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

# https://medium.com/@sara.khorram/inserting-python-dataframe-into-sql-table-increasing-the-speed-760a33db5ab5
method_6_insert_time_list = []
engine = create_engine(
    'mssql+pyodbc:///connection_string=', fast_executemany = True
)
dfs_list = []  # list of dayaframes
for df in dfs_list:
    df.to_sql('TableName', con = engine , if_exists = 'append', index = False)
