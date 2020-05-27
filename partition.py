import pandas as pd

from src.OntologyNode import findNodes, getDescendantNodes, getUpper
from src.ReadFD import reFD
from src.ReadJson import reJson

path='../../data/gdata/gdb.csv'
df=pd.read_csv(path,header=0)
df=pd.DataFrame(df,columns=['name','age','salary','location','department'])


print('Generalized Database')
print('')
print(df)


fd=reFD('../../data/sTest/testFd.csv')

location=reJson('../../data/gdata/ontology/city.json')
age=reJson('../../data/gdata/ontology/age.json')
department=reJson('../../data/gdata/ontology/department.json')

dic={'location':location,'age':age,'department':department}

# print(df)
# print('')
# print(df.ix[0][0])

class GeValue():
    def __init__(self, value,row,col,root):
        self.value = value
        self.row=row
        self.col=col
        self.root=root
        self.node=findNodes(root,value)
        self.height=node.getHeight()
        list=[self]
        self.partition=Partition(list)

class Partition():

    def __init__(self,list):
        self.list=list

    def getElements(self):
        return self.list


def merge(p1,p2):

    list=[]

    for i in p1.list:
        list.append(i)
    for k in p2.list:
        list.append(k)

    p=Partition(list)

    return p

def sortV(V):
    list={1:[],2:[],3:[],4:[],5:[]}    # default max level: 5 #
    for v in V:
        h=v.height
        list[h].append(v)
    return list



def maxV(V):
    for i in V:
        if len(V[i])>0:
            return V[i],i

print('--------------------------------')
print('--------------------------------')
print('')
print('First step, traverse the database, new GeValue()s, initialize the V and the P')
print('')

P=[]
columns=5
rows=len(df)
geColumn={1:'age',3:'location',4:'department'}
V={1:[],3:[],4:[]}

for y in geColumn:
    root=dic[geColumn[y]]
    for x in range(0,rows):
        value=df.ix[x][y]
        node=findNodes(root,value)
        if node.isLeaf()=='false':
            v=GeValue(value,x,y,root)
            V[y].append(v)
            P.append(v.partition)
    V[y]=sortV(V[y])

print('General Values V[1]:')
print(V[1])
print('')
print('General Values V[3]:')
print(V[3])
print('')
print('General Values V[4]:')
print(V[4])
print('')
print('Partitions P:')
print(P)
print('--------------------------------')
print('--------------------------------')

print('')
print('Second step, merge Partitions by FDs')
print('')
print(fd)

def doPartition(gex,gey,V,df,dic):

    xlist={}
    ylist={}
    for x in gex:
        x=x.split('*')
        xlist[int(x[0])]=dic[x[1]]
    #['1*age']

    for y in gey:
        y=y.split('*')
        ylist[int(y[0])]=dic[y[1]]
    #['3*location']

    for xx in xlist:   # 1：age

        max,rank=maxV(V[xx])
        while  rank <=5  :
            max=V[xx][rank]
            for mx in max:    #  <__main__.GeValue object at 0x11231a6a0>
                row=mx.row
                for yy in ylist: # 3：location
                    val_y=df.ix[row][yy]
                    root=ylist[yy]
                    node=findNodes(root,val_y)
                    upper=[]
                    upper=getUpper(root,node,upper)
                    for vy in V[yy]:   # vy=1,2,3,4,5
                        if len(V[yy][vy])>0: # current level is not an empty list
                            for v in V[yy][vy]:  # GeValue v
                                if v.value in upper:
                                    p=merge(mx.partition,v.partition)
                                    mx.partition=p
                                    v.partition=p
                                    for gv in p.list:
                                        gv.partition=p
            rank=rank+1
    return V

for f in fd:        #
    x=f
    y=fd[f]
    x=x.split(',')
    y=y.split(',')
    gex=[]
    gey=[]

    x_V={}
    y_V={}

    for xx in x:
        if len(xx)>1:
            gex.append(xx)

    for yy in y:
        if len(yy)>1:
            gey.append(yy)
    print(gex,gey)

    print('TEST!!!!!!!')

    V=doPartition(gex,gey,V,df,dic)

    for i in V[1][3]:

        print('this is '+i.value+'-----------------')
        p=i.partition
        e=p.getElements()
        for t in e:

            print(t.value)



















































