import pandas as pd

from src.data_cleaning import do_partition, consistent_partition, GeValue

from src.data_io import *

stepone = 'STEP ONE: Load the original database which only contains ground values'
path = '../data/test1/test40_general.csv'
df = pd.read_csv(path, header=0)
df = pd.DataFrame(df, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
age = read_json('../data/ontology/AGE.json')
symp = read_json('../data/ontology/SYMP.json')
gen = {1: 'male', 2: 'female'}
drug = {}
illness = {}

def reFD(filepath):

    file=open(filepath)
    Fd_Dic={}

    for row in file:
        fd=row.split(';')
        Fd_Dic[fd[0]]=fd[1]

    return Fd_Dic
fd = reFD('../data/test1/test40_fd_2.csv')
print(fd)

fdcolumns = [3, 4, 5]
rootdic = {'AGE': age, 'SYMP': symp}
path_ge = '../data/test1/test2partition.csv'
df_ge = pd.read_csv(path_ge, header=0)
df_ge = pd.DataFrame(df_ge, columns=['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS'])
P = []  # List of partitions
VPmap = {}  # Gevalue---->Partition
geColumn = {2: age, 3: symp}
columns = ['PID', 'GEN', 'AGE', 'SYMP', 'DRUG', 'ILLNESS']
do_partition(df_ge, geColumn, P, VPmap, fd)

print('Test Data ')
print('')
print(df_ge)
print('')
print('')
print('-----Do Partition-----')
IndexOfPartition = 0

for par in P:
    print('')
    print('No.' + str(IndexOfPartition) + ' of partition')
    print(par)
    rsl,rsl_informations=consistent_partition(df_ge, par, fd, columns)
    if rsl==True:
        print(rsl)
        print(rsl_informations)

    else:
        for info in rsl_informations:

            print(info.colIndex)
            print(info.rowIndex)

        for G in par.valuelist:
            if isinstance(G, GeValue) is True:
                if G.row in info.rowIndex:
                    if G.col in info.colIndex:
                        print(G.value)
                        tmp=G.frequency+1
                        G.frequency=tmp
                        print(G.frequency)

    IndexOfPartition = IndexOfPartition + 1

print(' ---Cleaning Result----')


for GV in VPmap:
    if isinstance(GV,GeValue) is True:
        print(GV.value)
        print(GV.frequency)
