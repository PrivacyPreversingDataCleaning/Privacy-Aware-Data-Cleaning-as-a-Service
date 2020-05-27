import pandas as pd
from IPython.display import display
from src.data_io import read_json
from src.data_cleaning import create_update_cell, apply_update_cell
from src.data_price import get_query_result, get_general_query_result, construct_constraints, calculate_weight, \
    calculate_price
from src.ontology_node import *

data_path = '../data/test1/5gdb.csv'
df_old = pd.read_csv(data_path, header=0)
print("original df:")
display(df_old)

qry1 = "select SYMP from df where GEN='female' ;"
qry2 = "select ILLNESS from df;"


result1 = get_query_result(qry1, df_old)
result2 = get_query_result(qry2, df_old)

# get the list of update
update = create_update_cell(df_old, 1)

df_new = apply_update_cell(df_old, update[0])

print("update is :", update[0])
print("new df:")

display(df_new)
result3 = get_query_result(qry2, df_new)

print('below is result for query 1 on old df')
display(result1)
print('below is result for query 2 on old df')
display(result2)
print('below is result for query 2 on new df')
display(result3)

## 以下测试 change the value in result, won't change the value in original df
# result1.iat[1,0] = 1
# display(result1)
# display(df_old)
#
# result4 = get_query_result(qry1, df_old)
# print (result1.equals(result4))
# print (result2.equals(result3))


# test getGequeryResult
result4 = get_query_result(qry1, df_old)
result4
symp =read_json('../data/ontology/SYMP.json')
n = value_to_node(symp, "fatigue")

geResult = get_general_query_result(qry1, df_old, symp, 1)

print("below is general query 1 on old df")
print(geResult)

display(df_old)
updateList = create_update_cell(df_old, 100)

queryList = [qry1, qry2]   # 用户 输入 的query

for u in updateList:
    print(str(u))
print(''.join(queryList))

pointList = [4, 10]     #query 1 和 query2 的价格， 用户先给一部分query定价， 系统根据这些再给新query定价

qry3 = "select DRUG from df;"  # 要计算价格的query

l = construct_constraints(updateList, df_old, queryList, pointList)
print("the constraints: ", l)

weights = calculate_weight(updateList, df_old, queryList, pointList)
print("the weights: ")

for index, w in enumerate(weights):
    print(index, ' : ', w)

price = calculate_price(qry3, updateList, df_old, queryList, pointList)
print('price for query', qry3, ': ', price)
