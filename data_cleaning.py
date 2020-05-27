import random
from collections import Counter, namedtuple

from src.data_io import tidy_df
from src.ontology_node import get_all_leafnodes
from src.ontology_node import get_ancestors
from src.ontology_node import value_to_node


class UpdateValue():
    '''
    Obj UpdateValue represents the basic tupple in the dataframe.
    '''

    def __init__(self, id, row, column, oldValue, newValue):
        self.id = id
        self.row = row
        self.column = column
        self.oldValue = oldValue
        self.newValue = newValue

    def set_row(self, row):
        self.row = row

    def get_row(self):
        return self.row

    def set_column(self, column):
        self.column = column

    def get_column(self):
        return self.column

    def set_newValue(self, value):
        self.newValue = value

    def get_newValue(self):
        return self.newValue

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def set_oldValue(self, value):
        self.oldValue = value

    def get_oldValue(self):
        return self.oldValue

    def __str__(self):
        r = ['id = ', str(self.id), ', row = ', str(self.row), ', column = ', str(self.column), ', oldValue = ',
             str(self.oldValue), ', newValue = ', str(self.newValue)]
        return ''.join(r)


class GeValue():
    def __init__(self, value, row, col, root):
        self.value = value
        self.row = row
        self.col = col
        self.root = root
        self.node = value_to_node(root, value)
        self.height = self.node.get_level()
        self.frequency=0


class GroundValue():
    def __init__(self, value, row, col, root):
        self.value = value
        self.row = row
        self.col = col
        self.root = root
        self.node = value_to_node(root, value)
        self.height = self.node.get_level()


class Partition:
    def __init__(self, valuelist):
        self.valuelist = valuelist


def create_update_cell(df, updateNum, attrNum=1):
    """
    given a domain dictionary of df, generate the update list.

    Args:
        df: the original database instance
        updateNum: the number of updates, the size of update list
        attrNum: the numer of attribute that will be updated, how many attributes will be changed in one update
    Return:
        a list of update obj
    """

    # generate the column domain value
    columnDomain = {}
    for column in df:
        columnDomain[column] = df[column].unique()

    # number of rows
    rowSize = df.shape[0]
    colSize = df.shape[1]

    updateList = []

    # give a random seed
    # random.seed(47)
    for i in range(updateNum):
        row = i % rowSize
        column = random.randint(0, colSize - 1)
        oldValue = str(df.iat[row, column]).strip()
        newValue = random.choice(columnDomain[df.columns[column]])

        u = UpdateValue(i, row, column, oldValue, newValue)
        updateList.append(u)
    return updateList


def apply_update_cell(df, updateCell):
    """
    apply the update cell to database instance
    Args:
        df: original database instance
        updateCell: the cell that will be applied
    """
    #     newInstance = df.copy(deep = True)
    df.ix[[updateCell.get_row()], [updateCell.get_column()]] = updateCell.get_newValue()
    return df


def rollback(df, updateCell):
    df.ix[[updateCell.get_row()], [updateCell.get_column()]] = updateCell.get_oldValue()
    return df


def getGeColumn(fddomain):
    dic = {}
    for column in fddomain:
        for index in column:
            for type in column[index]:
                if type == 'ontology':
                    dic[index] = column[index][type]

    return dic


def get_gevalue_acord_par(map, par):
    '''

    :param map:  全局字典，VPmap,保存GeValue对应partition的信息
    :param par:  一个partition
    :return:  得到对应此partition的GeValue list
    '''
    rsl = []

    for i in map:
        if map[i] == par:
            rsl.append(i)
    return rsl


def get_gevalue_acord_Y(V, y):
    for gev in V:
        if gev.row == y:
            return gev


def compare(gdb, columnX, columnY, GeValueX, Vlist, Plist, VPmaplist, geColumn):
    '''

    对一个fd左边的GeValue,求和它有关的所有
    :param gdb:
    :param column:
    :param GeValueX:
    :return:
    '''
    rsl = []
    df = gdb
    global P, VPmap
    P = Plist
    VPmap = VPmaplist
    root = geColumn[columnX]
    Xnode = GeValueX.node
    xspar = VPmap[GeValueX]
    uplist = []

    uplist = get_ancestors(root, Xnode)
    GeValuesRelatedToX = [GeValueX]  # 保存该轮需要放在一起的GeValues

    ## 首先判断对应的Y

    for GeValueY in Vlist[columnY]:
        if GeValueY.row == GeValueX.row:
            GeValuesRelatedToX.append(GeValueY)
    # Vlist[columnX].remove(GeValueX)

    ## 再判断在他层级之上的其他x以及对应的y

    for otherGeValuesInColumnX in Vlist[columnX]:

        if otherGeValuesInColumnX is not GeValueX:

            if otherGeValuesInColumnX.value in uplist or otherGeValuesInColumnX.value == GeValueX.value:
                GeValuesRelatedToX.append(otherGeValuesInColumnX)
                for otherGeValuesInColumnY in Vlist[columnY]:
                    if otherGeValuesInColumnY.row == otherGeValuesInColumnX.row:
                        GeValuesRelatedToX.append(otherGeValuesInColumnY)

    NewPartition = Partition([])
    #
    for GV in GeValuesRelatedToX:
        #
        GV_Partition = VPmap[GV]
        GV_Partition_Values = get_gevalue_acord_par(VPmap, GV_Partition)

        for thesevalues in GV_Partition_Values:
            VPmap[thesevalues] = NewPartition

        if GV_Partition in P:
            P.remove(GV_Partition)

    P.append(NewPartition)


def tidy_partitions(VPmap):
    P = []

    for Value in VPmap:

        if Value not in VPmap[Value].valuelist:
            VPmap[Value].valuelist.append(Value)

        if VPmap[Value] not in P:
            P.append(VPmap[Value])

    return P


def do_partition(df_ge, geColumn, P, VPmap, fd):
    # Initializing
    V = {}
    for c in geColumn:
        V[c] = []
    for y in geColumn:
        root = geColumn[y]
        for x in range(0, len(df_ge)):
            value = str(df_ge.ix[x][y])
            node = value_to_node(root, value)
            if node.is_leaf() == False:
                v = GeValue(value, x, y, root)
                V[y].append(v)
                p = Partition([v])
                P.append(p)
                VPmap[v] = p

    for f in fd:
        x = f
        y = fd[f]
        x = x.split(',')
        y = y.split(',')
        gex = []
        gey = []

        for xx in x:
            if len(xx) > 1:
                gex.append(int(xx[0]))

        for yy in y:
            if len(yy) > 1:
                gey.append(int(yy[0]))

        for row in gex:
            for current_vx in V[row]:
                for column in gey:
                    compare(df_ge, row, column, current_vx, V, P, VPmap, geColumn)  # vx is the current GeValue

    NP = tidy_partitions(VPmap)

    for par in NP:

        for gv in par.valuelist:

            if gv.node.is_leaf() is False and gv.col in gex:
                col = gv.col
                root = gv.root
                node = gv.node

                leavesUnderGeValue = get_all_leafnodes(root, node)
                for row in range(0, len(df_ge)):
                    if df_ge.ix[row][col] in leavesUnderGeValue:
                        groundvalue = GroundValue(df_ge.ix[row][col], row, col, root)
                        par.valuelist.append(groundvalue)
                        VPmap[groundvalue] = par


def check_consistency(df, fd, columns):
    fd_violation_index={}

    for f in fd:  # 0,1*age;3*location
        x = f
        y = fd[f]
        x = x.split(',')
        y = y.split(',')
        leftcols = []
        rightcols = []
        for c in x:
            if len(c) > 1:
                leftcols.append(int(c[0]))
        for c in y:
            if len(c) > 1:
                rightcols.append(int(c[0]))
        cols = []
        indexcols = []
        for c in leftcols:
            cols.append(columns[c])
            indexcols.append(c)
        for c in rightcols:
            cols.append(columns[c])
            indexcols.append(c)
        cols_groups = df.groupby(cols)
        # # groups 返回 dict (key = group的key, value = row index)
        groups_keys = cols_groups.groups.keys()
        # # t[:-1] 是 key1, key2, 没有key3, 也就是FD 的 X
        y_len = len(rightcols)
        c = Counter([t[:-y_len] for t in groups_keys])
        indexes=[]
        fd_violation_index[f] = []
        for group, index in cols_groups.groups.items():
            #     # 对于k1k2 ->k3 来说, 如果k1,k2,k3的所有group中
            #     # k1,k2 对应的group多过 1, 说明 k3 有不一样的
            # print(c.get(group[:-y_len]))

            if c.get(group[:-y_len]) > 1:
                # print(group)
                # print(index)
                indexes.append(index)

            if len(indexes)>1:

                fd_violation_index[f].append(indexes)
    for f in fd_violation_index:

        print(fd_violation_index)

        if len(fd_violation_index[f])>0:
            return False,fd_violation_index
        else:
            return True, fd_violation_index

    # return True,fd_violation_index


def consistent_partition(df, partition, fd, columns):
    '''
    先通过深度复制生成partiondf，对其进行如下更改：
        对所有不存在该partion元素的行，行内所有列值改为0
    :param df:
    :param partition:
    :param fd:
    :return:
    '''
    import copy
    partitiondf = copy.deepcopy(df)
    tidedPartitionDf = tidy_df(partitiondf, partition, columns)

    violate_information=namedtuple('Violation', ['colIndex','rowIndex'])
    violate_informations=[]

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('DF Before Replacing the Update Ground Value')
    print(tidedPartitionDf)
    print('+++++++++++++++++++++++++++++++++++++++++++')
    id = 0
    All_Updates = []  ## 所有的 update


    for GeV in partition.valuelist:

        if isinstance(GeV, GeValue) is True:

            GeValue_Updates = []
            GroundValuesUnderGeValue = get_all_leafnodes(GeV.root, GeV.node)
            row = GeV.row
            col = GeV.col
            oldValue = GeV.value

            for leaf in GroundValuesUnderGeValue:
                newValue = leaf
                updatevalue = UpdateValue(id, row, col, oldValue, newValue)
                GeValue_Updates.append(updatevalue)
                id = id + 1

            id = id + 1
            All_Updates.append(GeValue_Updates)

    import itertools

    UpdateCombinations = list(itertools.product(*All_Updates))

    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print('All the Combinations can be Applied On The DF:')

    for OnePairOfCombinations in UpdateCombinations:
        testOutputShowingInformationOfCell=[]

        for UpdateCell in OnePairOfCombinations:
            testOutputShowingInformationOfCell.append(UpdateCell)

        print('')

        for cell in testOutputShowingInformationOfCell:

            print('[Row]:'+str(cell.row),'[Col]:'+str(cell.column),'[Old Value]:'+str(cell.oldValue),'[New Value]:'+str(cell.newValue))

    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    violate_rows=[]

    for OnePairOfCombinations in UpdateCombinations:
        testOutputShowingInformationOfCell = []
        for UpdateCell in OnePairOfCombinations:
            apply_update_cell(tidedPartitionDf, UpdateCell)
            testOutputShowingInformationOfCell.append(UpdateCell)

        print('+++++++++++++++++++++++++++++++++++++++++++')
        print('The Updating Combination Applied On The DF')
        print('')

        for cell in testOutputShowingInformationOfCell:

            print('[Row]:' + str(cell.row), '[Col]:' + str(cell.column), '[Old Value]:' + str(cell.oldValue),
                  '[New Value]:' + str(cell.newValue))

        print('')
        print('DF After Replacing the Update Ground Value')
        print(tidedPartitionDf)
        print('+++++++++++++++++++++++++++++++++++++++++++')


        check_rsl=check_consistency(tidedPartitionDf, fd, columns)

        if check_rsl[0] == True:
            print(violate_information)
            return True,violate_informations
        else:
            violate_rows.append(check_rsl[1])

    print('HERE SHOULD ADD FLASE FREQUENCY ON THE UPDATE CELL')


    fd_id=0

    for f in fd:

        cols=[]
        for c in f.split(','):
            cols.append(int(c.split('*')[0]))
        for cy in fd[f].split(','):
            cols.append(int(cy.split('*')[0]))
        print(cols)

        violates_set=[]

        for vr in violate_rows:
            if vr[f] not in violates_set:
                violates_set.append(vr[f])

        violates_rows=[]


        for vs in violates_set:
            for l in vs:
                lrows=[]
                for int_index in l:
                    lrows.append(int_index[0])

                if len(violates_rows)==0:
                    violates_rows=lrows

                else:
                    tmp = [val for val in lrows if val in violates_rows]
                    violates_rows=tmp

        print(violates_rows)

        v= violate_information(cols, violates_rows)
        violate_informations.append(v)


    return False,violate_informations