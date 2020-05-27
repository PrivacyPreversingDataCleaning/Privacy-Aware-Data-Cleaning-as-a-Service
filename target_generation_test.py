import pandas as pd
from data_io import read_json, read_fd

from src.target_generation import generate_target

age = read_json('../data/test1/AGE.json')
symp = read_json('../data/test1/SYMP.json')
gen = {1: 'male', 2: 'female'}
drug = {}
illness = {}
fd = read_fd('../data/test1/test40_fd.csv')
fdcolumns = [2, 3]
rootdic = {'AGE': age, 'SYMP': symp}
path_master = '../data/test1/test2matching.csv'
df_master = pd.read_csv(path_master, header=0)
df_master = pd.DataFrame(df_master, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
P = []  # List of partitions
VPmap = {}  # Gevalue---->Partition
geColumn = {2: age, 3: symp}
columns = ['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS']


print(generate_target(df_master,fdcolumns,error_num=3,percentage=0.8))