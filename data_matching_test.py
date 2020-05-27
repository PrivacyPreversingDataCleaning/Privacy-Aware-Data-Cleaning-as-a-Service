import pandas as pd
from data_cleaning import GeValue
from data_io import *

from src.data_matching import find_matches

age = read_json('../data/test1/AGE.json')
symp = read_json('../data/test1/SYMP.json')
gen = {1: 'male', 2: 'female'}
drug = {}
illness = {}
fd = read_fd('../data/test1/test40_fd.csv')
fdcolumns = [3, 4, 5]
rootdic = {'AGE': age, 'SYMP': symp}
path_master = '../data/test1/test2matching.csv'
df_master = pd.read_csv(path_master, header=0)
df_master = pd.DataFrame(df_master, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
P = []  # List of partitions
VPmap = {}  # Gevalue---->Partition
geColumn = {2: age, 3: symp}
columns = ['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS']
row2 = pd.Series(data=[522, 'female', '[0-10]', 'physical', 'Advil', 'leukemia'],
          index=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])




