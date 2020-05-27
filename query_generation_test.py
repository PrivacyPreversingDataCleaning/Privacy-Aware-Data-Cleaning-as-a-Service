import pandas as pd
from data_cleaning import GeValue
from data_io import read_json, read_fd
from data_price import get_query_result
from query_generation import matched_df_generator, query_generator

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
ge_value=GeValue('[0-10]', 11, 2, age)

violation_GeValue=[]
violation_GeValue.append(ge_value)
query_df_list=query_generator(violation_GeValue,df_master,geColumn)

for query_df in query_df_list:

    qry=query_df[0]
    df=query_df[1]
    print(qry)
    print(df)
    print(get_query_result(qry, df))