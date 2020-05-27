"""
Author: Yu Huang
Python Version: 3.5
"""
import pandas as pd
from data_cleaning import do_partition, consistent_partition, GeValue
from data_io import read_fd, read_json
from data_price import calculate_price, get_query_result

from src.query_generation import query_generator

# target = generate_target()
path = '../data/test1/test40_general.csv'
df = pd.read_csv(path, header=0)
df = pd.DataFrame(df, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
age = read_json('../data/test1/AGE.json')
symp = read_json('../data/test1/SYMP.json')
gen = {1: 'male', 2: 'female'}
drug = {}
illness = {}
fd = read_fd('../data/test1/test40_fd.csv')
fdcolumns = [3, 4, 5]
rootdic = {'AGE': age, 'SYMP': symp}
path_ge = '../data/test1/test2partition.csv'
df_ge = pd.read_csv(path_ge, header=0)
df_ge = pd.DataFrame(df_ge, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
P = []  # List of partitions
VPmap = {}  # Gevalue---->Partition
geColumn = {2: age, 3: symp}
columns = ['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS']

# do_partition(df_ge, geColumn, P, VPmap, fd)



#

def rank_error_cells(df_target, fds, ontology_roots):
    """
    生成排序后的 error cells
    :param df_target: 
    :param fds: a list of fd
    :param ontology_roots: 相对应的 ontology 的 root, 可以是dict或者list
    dict 的话, key是 column_index, value 是 root object
    :return: a list of error object, 
    """
    do_partition(df_target, geColumn, P, VPmap, fd)
    for par in P:
        rsl, rsl_informations = consistent_partition(df_ge, par, fd, columns)
        if rsl == False:
            for info in rsl_informations:
                for G in par.valuelist:
                    if isinstance(G, GeValue) is True:
                        if G.row in info.rowIndex:
                            if G.col in info.colIndex:
                                tmp = G.frequency + 1
                                G.frequency = tmp
    GeValues_list=[]
    for value in VPmap:
        if isinstance(value,GeValue):
            GeValues_list.append(value)

    sorted_GeValues_list = sorted(GeValues_list, key=lambda GeValue: GeValue.frequency,reverse=True)

    return sorted_GeValues_list

ranked_error_cells = rank_error_cells(df_ge,fd,rootdic)

for c in ranked_error_cells:
    print(c.value)
    print(c.frequency)


print('----------------------------=================-----------------------')
query_df_list=query_generator(ranked_error_cells,df_ge,geColumn)

for query_df in query_df_list:
    qry=query_df[0]
    df=query_df[1]
    print(qry)
    print(df)# 计算query的price, 需要 supportset(updateList), original database(df_old)
    print(get_query_result(qry, df))# queryList, pointList, 用户指定的query和price point



'''
TODO

match df : match 不到


'''