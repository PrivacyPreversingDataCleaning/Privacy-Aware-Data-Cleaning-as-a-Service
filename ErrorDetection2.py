from src.OntologyNode import getFamily, findNodes
from src.ReadJson import reJson


def ErrorDetect(MasterData,data, fd_dic):

    datadic={}
    violations = {}
    for fd in fd_dic:
        X = fd
        Y = fd_dic[fd]

        vio_key = str(X) + '-' + str(Y)

        violations[vio_key] = ErrorDetect_perFD(MasterData,data, X, Y)

    return violations


def ErrorDetect_perFD(MasterData,data,X,Y):

    X, Y = X.split(','), Y.split(',')
    datadic= datadic_fd(MasterData,X,Y)   #{fd:{strX:[index,listY]}
    violations = []

    for i in range(0, len(data)):

        tuple=data[i]

        # save into the data dic of the fd

        strX = ''   # left part of fd

        for x in X:
            strX = strX + str(tuple[int(x)])


        tempY=[]

        for y in Y:
            y = y.split('*')
            tempY.append(tuple[int(y[0])])


        if strX in datadic:

            listA=datadic[strX]

            check=equals(listA,tempY)

            if check is 'false':

                vio=[i,strX,tempY]

                violations.append(vio)
        else:

            vio = [i, strX, tempY]

            violations.append(vio)


    return violations

def equals(listA,listB):   #listA is the listY in the data dic

    tag='true'

    if len(listA)==len(listB):

        for i in range(0,len(listA)):
            if listB[i]!=listA[i] and listB[i] not in listA[i]:
                tag='false'
                break

    return tag

def datadic_fd(MasterData,X,Y):

    # X, Y = X.split(','), Y.split(',')
    datadic = {}
    for i in range(0, len(MasterData)):

        tuple=MasterData[i]

        # save into the data dic of the fd

        strX = ''   # left part of fd
        listY = []  # right part of fd, because of the possibility of Ontology, so use list to combine

        for x in X:
            strX = strX + str(tuple[int(x)])

        for y in Y:

            y = y.split('*')

            if len(y) > 1:
                # pattern of y*tree

                root = reJson('../data/lTest/city.json')    #path according to the y[1]

                listY.append(getFamily(y[1], findNodes(root, tuple[int(y[0])])))

            else:

                listY.append(tuple[int(y[0])])

            datadic[strX]=listY

    return datadic

