import pandas as pd


DATA_ROOT = './data/'
COL_NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']


 
# Read the train and test data
df = pd.read_fwf(DATA_ROOT+'Iris_train.dat')
df2 = pd.read_fwf(DATA_ROOT+'Iris_test.dat')

# Rename the columns
df.columns = COL_NAMES
df2.columns = COL_NAMES

iris = pd.load_csv()