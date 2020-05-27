import random

import pandas as pd

from OntologyNode import findNodes, getDescendantNodes, getUpper, getRandom
from query.gequery import gequerry
from utility.ReadFD import reFD
from utility.ReadJson import reJson


path='../data/test1/1000gdb.csv'
df=pd.read_csv(path,header=0)
df=pd.DataFrame(df,columns=['PID','GEN','AGE','SYMP','DRUG','ILLNESS'])
age=reJson('../data/test1/age.json')
symp=reJson('../data/test1/symp.json')
gen={1:'male',2:'female'}
drug={}
illness={}
fd=reFD('../data/test1/fd.csv')
fdcolumns=[3,4,5]
domain=[{0:{'number':'1,1000'}},
        {1:{'value':gen}},
        {2:{'ontology':age}},
       ]


def fdpattern(df,fdcolumns):
    patternlist=[]
    length=len(df)
    for l in range(0,length):
        t=[]
        for c in fdcolumns:
            t.append(df.ix[l][c])
        if t not in patternlist:
            patternlist.append(t)
    return patternlist

class update():
    '''

    TODO:

    '''


def update(amount,df,domain,fdcolumns,pattern):

    update=[]

    for i in range(0,amount):

        randomrow=random.randint(0,len(df)-1)     #### next version, every row at least one tupple has to be updated
        tag=random.randint(0,1)
        if tag==0:  ##更改非fd列
            max=len(domain)-1
            rand=random.randint(0,max)
            randomcol=domain[rand]

            for c in randomcol:
                col=c
                coldom=randomcol[c]


            for d in coldom:
                if d=='ontology':
                    root=coldom[d]
                    oldvalue=str(df.ix[randomrow][col])
                    choice=getRandom(root, oldvalue)
                    record=[randomrow,col,choice]

                    update.append(record)

                if d=='value':
                    dd=coldom[d]
                    max=0
                    for i in dd:
                        max=max+1
                    choice=dd[random.randint(1,2)]
                    record=[randomrow,col,choice]

                    update.append(record)

                if d=='number':
                    dd = coldom[d]
                    rg=dd.split(',')
                    min=int(rg[0])
                    max=int(rg[1])
                    choice=random.randint(min,max)
                    record=[randomrow,col,choice]

                    update.append(record)

        if tag==1:
            pattern = random.sample(pattern, 1)[0]
            for i in range(0,len(pattern)):
                record=[randomrow,fdcolumns[i],pattern[i]]
                update.append(record)

    return update

def updateEveryRow(amount,df,domain,fdcolumns,pattern):

    updates=[]

    for i in range(0,amount):

        for randomrow in range(0,len(df)-1):
            tag=random.randint(0,1)
            if tag==0:  ##更改非fd列
                update = []
                max=len(domain)-1
                rand=random.randint(0,max)
                randomcol=domain[rand]

                for c in randomcol:
                    col=c
                    coldom=randomcol[c]

                for d in coldom:
                    if d=='ontology':
                        root=coldom[d]
                        oldvalue=str(df.ix[randomrow][col])
                        choice=getRandom(root, oldvalue)
                        record=[randomrow,col,choice]
                        update.append(record)

                    if d=='value':
                        dd=coldom[d]
                        max=0
                        for i in dd:
                            max=max+1
                        choice=dd[random.randint(1,2)]
                        record=[randomrow,col,choice]
                        update.append(record)

                    if d=='number':
                        dd = coldom[d]
                        rg=dd.split(',')
                        min=int(rg[0])
                        max=int(rg[1])
                        choice=random.randint(min,max)
                        record=[randomrow,col,choice]
                        update.append(record)

                updates.append(update)

            if tag == 1:
                update = []
                pat = random.sample(pattern, 1)[0]
                l=len(pat)
                for t in range(0, len(pat)):
                    record = [randomrow, fdcolumns[t], pat[t]]
                    # print(record)
                    update.append(record)

                updates.append(update)

    return updates

def updateGeneral(amount,df,domain,fdcolumns,pattern):
    updates=updateEveryRow(amount,df,domain,fdcolumns,pattern)
    max=random.randint(100,1000)

    for t in range(0,max):

        updates.append(update(amount,df,domain,fdcolumns,pattern))

    return updates

print('---------TEST UPDATE----------')
pattern=fdpattern(df,fdcolumns)
updates=updateGeneral(1,df,domain,fdcolumns,pattern)
for r in updates:
     print(r)

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

def gequerry(query,exp_level,rootdic):
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
            gersl.append(geselect(rootdic[col],i,exp_level))
    return gersl

query="select SYMP from df where GEN='female' ;"
exp_level=2
rootdic={'AGE':age,'SYMP':symp}

print('*******')
print('')
print('*******')

print('---------TEST QUERRY----------')
print(gequerry(query,exp_level,rootdic))


print('*******')
print('')
print('*******')
print('--------Test APPLY and Filter------------')

def apply(df,update):
    master=[]
    for u in update:

        row=u[0]
        col=df.columns[u[1]]

        original=[row,u[1],df.ix[row, col]]

        value=u[2]
        df.ix[row, col] = value
        master.append(original)

    return df,master

class supportGDB():
    '''
    TODO: combing df and update
    '''



update=update(1,df,domain,fdcolumns,pattern)
apply(df,update)


def filter(df,updates,query,exp_level,rootdic):

    rsl=[]

    correct=gequerry(query,exp_level,rootdic)
    for update in updates:
        df,maintain=apply(df,update)
        te=gequerry(query,exp_level,rootdic)
        if te==correct:
            # print('true')
            rsl.append(update)
        # else:
        #     print('false')
        df=apply(df,maintain)[0]

    return rsl
print('original number of updates:'+str(len(updates)))
rsl=filter(df,updates,query,exp_level,rootdic)

print('after filtering:'+str(len(rsl)))

