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

coldic={1:'age',3:'location',4:'department'}
print('')
# print(df.ix[0][0])

class GeValue():
    def __init__(self, value,row,col,root):
        self.value = value
        self.row=row
        self.col=col
        self.root=root
        self.node=findNodes(root,value)
        self.height=self.node.getHeight()

class Partition():

    def __init__(self,list):
        self.list=list

def getGeValueAcordPar(map,par):

    rsl=[]

    for i in map:
        if map[i]==par:
            rsl.append(i)
    return rsl

def getGeValueAcordY(V,y):

    for gev in V:
        if gev.row==y:
            return gev


def compare(gdb,columnX,columnY,GeValueX,Vlist,Plist,VPmaplist):
    '''
    :param gdb:
    :param column:
    :param GeValueX:
    :return:
    '''
    rsl=[]
    df=gdb
    global V,P,VPmap
    V=Vlist
    P=Plist
    VPmap=VPmaplist
    root=geColumn[columnX]
    Xnode=GeValueX.node
    xspar = VPmap[GeValueX]
    delist = [GeValueX.value]
    delist = getDescendantNodes(root, Xnode, delist)   # value list
    # print(delist)
    for row in range(0,len(df)):
        xdata=df.ix[row][columnX]
        # print(xdata+'++++++')
        if xdata in delist:
            ydata=df.ix[row][columnY]
            rooty=geColumn[columnY]
            heighty = findNodes(rooty, ydata).getHeight()
            # print(ydata)
            # print(heighty)
            for GeValueY in V[columnY]:   # find the equal one in right side
                # print(GeValueY.value)
                # if GeValueY.value==ydata:
                # print(findNodes(rooty,GeValueY.value).getHeight())
                if findNodes(rooty,GeValueY.value).getHeight()<=heighty:
                    yspar=VPmap[GeValueY]
                    # print(yspar)
                    p=Partition([])
                    # print(VPmap[GeValueX],VPmap[GeValueY])
                    if xspar in P:
                        GVS = getGeValueAcordPar(VPmap, xspar)
                        for gs in GVS:
                            VPmap[gs]=p
                        P.remove(xspar)
                    if yspar in P:
                        GVS = getGeValueAcordPar(VPmap, yspar)
                        # print(GVS)
                        for gs in GVS:
                            VPmap[gs] = p
                        P.remove(yspar)

                    xspar=VPmap[GeValueX]
                    # print(VPmap[GeValueX], VPmap[GeValueY])
                    P.append(p)
P=[]      # list of partitions

VPmap={}   # GeValue -----> Partition

columns=5
rows=len(df)
geColumn={1:age,3:location,4:department}    # column ---> root
V={1:[],3:[],4:[]}  #save every column's GeValue

# Initializing
for y in geColumn:
    root=geColumn[y]
    for x in range(0,rows):
        value=df.ix[x][y]
        node=findNodes(root,value)
        if node.isLeaf()=='false':
            v=GeValue(value,x,y,root)
            V[y].append(v)
            p=Partition([v])
            P.append(p)
            VPmap[v]=p
# Traverse
print(P)
print('--')


for f in fd:    # 0,1*age;3*location
    x=f
    y=fd[f]
    x=x.split(',')
    y=y.split(',')
    gex=[]
    gey=[]

    for xx in x:
        if len(xx)>1:
            gex.append(int(xx[0]))

    for yy in y:
        if len(yy)>1:
            gey.append(int(yy[0]))

    for a in gex:
        xroot=geColumn[a]
        for vx in V[a]:
            # print(vx.value)

            # print('its compare')
            for b in gey:
                compare(df, a,b, vx,V,P,VPmap)
                # print('--------------------------------------------------------')
print(P)

print('****')


for ll in VPmap:
    print(ll.value)
    print(VPmap[ll])


