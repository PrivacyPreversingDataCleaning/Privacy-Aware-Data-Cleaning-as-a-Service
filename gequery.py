import pandas as pd

from src.OntologyNode import findNodes, getLeaves
from src.ReadJson import reJson

print('---initiallizing the gdb-----')

print('')
path='../../data/gdata/gdb.csv'
df=pd.read_csv(path,header=0)
df=pd.DataFrame(df,columns=['name','age','salary','location','department'])
print(df)

print('')

# initializing the ontology tree
location=reJson('../../data/gdata/ontology/city.json')
age=reJson('../../data/gdata/ontology/age.json')
dic={'location':location,'age':age}

def geselect(root,value,exp_level):
    node=findNodes(root,value)
    if node is not None:
        level=node.getHeight()
        diff=exp_level-level
        if diff==0:
            return value
        if diff<0:
            diff=-diff
            temp=node
            for i in range(0,diff):
                temp=temp.getParent()
                va=temp.value
            return temp.value
        if diff>0:
            temp=node.getChildren()
            for i in range(1,diff):
                temp2=[]
                for c in temp:
                    temp2=temp2+c.getChildren()
                temp=temp2
            rsl=[]
            for node in temp:
                rsl.append(node.value)
            return rsl

def gequerry(query,exp_level):

    from pandasql import sqldf
    pysqldf = lambda q: sqldf(q, globals())
    rsl=pysqldf(query)
    columns=[]
    for c in rsl.columns:
        columns.append(str(c))
    for col in columns:
        a=rsl[col]
        gersl=[]
        for i in a:
            gersl.append(geselect(dic[col],i,exp_level))
    return gersl




query="select location from df where age='[31-40]' ;"
exp_level=1

print('-------------------')
print('Query Test: select location where age=[31-40], expected level is 1')
print('')
print('RESULT: ')
print(gequerry(query,exp_level))


print('')
print('-------------------')
print('Test 2: According to the result value, get the information of ontology family')

print('Test value: age: [31-40]')
tv='[31-40]'

node=findNodes(age,tv)

print('NODE:')
print(node)
print('')
print('FAMILY')
from src.OntologyNode import getFamily
family=getFamily(age,node)
print(family)
print('')
print('LEAVES')
print('')
leaves=getLeaves(age,node)
print(leaves)







