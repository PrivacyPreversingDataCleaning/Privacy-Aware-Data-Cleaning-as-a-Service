from src.data_io import *

jpath = '../data/test1/SYMP.json'

root = read_json(jpath)

for i in root.get_children():
    print(i.value)

l = root.get_leafnodes()
for j in l:
    print(j)
