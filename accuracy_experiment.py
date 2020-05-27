import configparser
import copy
import os
import random

import pandas as pd

from collections import Iterable

import time


def _load_json(filepath):
    '''
    读取json文件
    :param filepath: 测
    :return: 数据字典 dic
    '''
    import json
    fp = filepath
    f = open(fp)
    dic = json.load(f)
    return dic


def _dict_to_tree(dic):
    """
    convert a dict to ontology node
    :param dic:
    :return:
    """
    current_node = OntologyNode(dic['value'])
    current_node_children = dic['children']
    if current_node_children is not None:
        for child in current_node_children:
            child_node = _dict_to_tree(child)
            child_node.set_parent(current_node)
            current_node.add_child(child_node)
    return current_node


def read_json(filepath):
    '''
    整合上述方法，返回以root为根的树
    example：
    country=reJson('../data/lTest/country.json')
    :param filepath:
    :return:
    '''
    return _dict_to_tree(_load_json(filepath))


def load_ontology_dict(folder_path):
    '''
    file.name（column's name）作为key
    :param folder_path: 包含 json files的文件夹路径
    :return:
    '''
    ontology_dict = scan_ontology_folder(folder_path)
    for tree in ontology_dict:
        file_path = ontology_dict[tree]
        ontology_dict[tree] = read_json(file_path)
    return ontology_dict


def tidy_df(partitiondf, partition, columns):
    '''

    Make rest ZERO
    :param partitiondf:
    :return:
    '''
    GeValueRows = []

    for Values in partition.valuelist:
        GeValueRows.append(Values.row)
    for row in range(0, len(partitiondf)):
        if row not in GeValueRows:
            for col in range(0, len(columns)):
                partitiondf.ix[row, col] = 0
    return partitiondf


def read_config(path):
    """
    read the config file to load parameter and files

    :param path: the path of config file
    :return: a config dict, which can be used as config['DATA']['master']
    """
    config = configparser.ConfigParser()
    config.read(path)
    # config.sections()
    return config


class Partition(object):
    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point, use index to avoid negative values
    self.high: higher point, use index to avoid negative values
    self.allow: show if partition can be split on this QI
    """

    def __init__(self, data, low, high):
        """
        split_tuple = (index, low, high)
        """
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allow = [1] * QI_LEN

    def add_record(self, record, dim):
        """
        add one record to member
        """
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        """
        add multiple records (list) to partition
        """
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        """
        return number of records
        """
        return len(self.member)


def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    d_order = QI_ORDER[index]
    width = float(d_order[partition.high[index]]) - float(d_order[partition.low[index]])
    return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    """
    chooss dim with largest norm_width from all attributes.
    This function can be upgraded with other distance function.
    """
    max_width = -1
    max_dim = -1
    for dim in range(QI_LEN):
        if partition.allow[dim] == 0:
            continue
        norm_width = get_normalized_width(partition, dim)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = dim
    if max_width > 1:
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition, return splitVal
    """
    # use frequency set to get median
    frequency = frequency_set(partition, dim)
    splitVal = ''
    nextVal = ''
    value_list = frequency.keys()
    value_list.sort(cmp=cmp_str)
    total = sum(frequency.values())
    middle = total / 2
    if middle < GL_K or len(value_list) <= 1:
        try:
            return ('', '', value_list[0], value_list[-1])
        except IndexError:
            return ('', '', '', '')
    index = 0
    split_index = 0
    for i, qi_value in enumerate(value_list):
        index += frequency[qi_value]
        if index >= middle:
            splitVal = qi_value
            split_index = i
            break
    else:
        print("Error: cannot find splitVal")
    try:
        nextVal = value_list[split_index + 1]
    except IndexError:
        # there is a frequency value in partition
        # which can be handle by mid_set
        # e.g.[1, 2, 3, 4, 4, 4, 4]
        nextVal = splitVal
    return (splitVal, nextVal, value_list[0], value_list[-1])


def anonymize_strict(partition):
    """
    recursively partition groups until not allowable
    """
    allow_count = sum(partition.allow)
    # only run allow_count times
    if allow_count == 0:
        RESULT.append(partition)
        return
    for index in range(allow_count):
        # choose attrubite from domain
        dim = choose_dimension(partition)
        if dim == -1:
            print("Error: dim=-1")
            pdb.set_trace()
        (splitVal, nextVal, low, high) = find_median(partition, dim)
        # Update parent low and high
        if low is not '':
            partition.low[dim] = QI_DICT[dim][low]
            partition.high[dim] = QI_DICT[dim][high]
        if splitVal == '' or splitVal == nextVal:
            # cannot split
            partition.allow[dim] = 0
            continue
        # split the group from median
        mean = QI_DICT[dim][splitVal]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = QI_DICT[dim][nextVal]
        lhs = Partition([], partition.low, lhs_high)
        rhs = Partition([], rhs_low, partition.high)
        for record in partition.member:
            pos = QI_DICT[dim][record[dim]]
            if pos <= mean:
                # lhs = [low, mean]
                lhs.add_record(record, dim)
            else:
                # rhs = (mean, high]
                rhs.add_record(record, dim)
        # check is lhs and rhs satisfy k-anonymity
        if len(lhs) < GL_K or len(rhs) < GL_K:
            partition.allow[dim] = 0
            continue
        # anonymize sub-partition
        anonymize_strict(lhs)
        anonymize_strict(rhs)
        return
    RESULT.append(partition)


def anonymize_relaxed(partition):
    """
    recursively partition groups until not allowable
    """
    if sum(partition.allow) == 0:
        # can not split
        RESULT.append(partition)
        return
    # choose attrubite from domain
    dim = choose_dimension(partition)
    if dim == -1:
        print("Error: dim=-1")
        pdb.set_trace()
    # use frequency set to get median
    (splitVal, nextVal, low, high) = find_median(partition, dim)
    # Update parent low and high
    if low is not '':
        partition.low[dim] = QI_DICT[dim][low]
        partition.high[dim] = QI_DICT[dim][high]
    if splitVal == '':
        # cannot split
        partition.allow[dim] = 0
        anonymize_relaxed(partition)
        return
    # split the group from median
    mean = QI_DICT[dim][splitVal]
    lhs_high = partition.high[:]
    rhs_low = partition.low[:]
    lhs_high[dim] = mean
    rhs_low[dim] = QI_DICT[dim][nextVal]
    lhs = Partition([], partition.low, lhs_high)
    rhs = Partition([], rhs_low, partition.high)
    mid_set = []
    for record in partition.member:
        pos = QI_DICT[dim][record[dim]]
        if pos < mean:
            # lhs = [low, mean)
            lhs.add_record(record, dim)
        elif pos > mean:
            # rhs = (mean, high]
            rhs.add_record(record, dim)
        else:
            # mid_set keep the means
            mid_set.append(record)
    # handle records in the middle
    # these records will be divided evenly
    # between lhs and rhs, such that
    # |lhs| = |rhs| (+1 if total size is odd)
    half_size = len(partition) / 2
    for i in range(half_size - len(lhs)):
        record = mid_set.pop()
        lhs.add_record(record, dim)
    if len(mid_set) > 0:
        rhs.low[dim] = mean
        rhs.add_multiple_record(mid_set, dim)
    # It's not necessary now.
    # if len(lhs) < GL_K or len(rhs) < GL_K:
    #     print "Error: split failure"
    # anonymize sub-partition
    anonymize_relaxed(lhs)
    anonymize_relaxed(rhs)


def init(data, k, QI_num=-1):
    """
    reset global variables
    """
    global GL_K, RESULT, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    GL_K = k
    RESULT = []
    # static values
    QI_DICT = []
    QI_ORDER = []
    QI_RANGE = []
    att_values = []
    for i in range(QI_LEN):
        att_values.append(set())
        QI_DICT.append(dict())
    for record in data:
        for i in range(QI_LEN):
            att_values[i].add(record[i])
    for i in range(QI_LEN):
        value_list = list(att_values[i])
        value_list.sort(cmp=cmp_str)
        QI_RANGE.append(float(value_list[-1]) - float(value_list[0]))
        QI_ORDER.append(list(value_list))
        for index, qi_value in enumerate(value_list):
            QI_DICT[i][qi_value] = index


def mondrian(data, k, relax=False, QI_num=-1):
    """
    Main function of mondrian, return result in tuple (result, (ncp, rtime)).
    data: dataset in 2-dimensional array.
    k: k parameter for k-anonymity
    QI_num: Default -1, which exclude the last column. Othewise, [0, 1,..., QI_num - 1]
            will be anonymized, [QI_num,...] will be excluded.
    relax: determine use strict or relaxed mondrian,
    Both mondrians split partition with binary split.
    In strict mondrian, lhs and rhs have not intersection.
    But in relaxed mondrian, lhs may be have intersection with rhs.
    """
    init(data, k, QI_num)
    result = []
    data_size = len(data)
    low = [0] * QI_LEN
    high = [(len(t) - 1) for t in QI_ORDER]
    whole_partition = Partition(data, low, high)
    # begin mondrian
    start_time = time.time()
    if relax:
        # relax model
        anonymize_relaxed(whole_partition)
    else:
        # strict model
        anonymize_strict(whole_partition)
    rtime = float(time.time() - start_time)
    # generalization result and
    # evaluation information loss
    ncp = 0.0
    dp = 0.0
    for partition in RESULT:
        rncp = 0.0
        for index in range(QI_LEN):
            rncp += get_normalized_width(partition, index)
        rncp *= len(partition)
        ncp += rncp
        dp += len(partition) ** 2
        for record in partition.member[:]:
            for index in range(QI_LEN):
                if isinstance(record[index], int):
                    if partition.low[index] == partition.high[index]:
                        record[index] = '%d' % (QI_ORDER[index][partition.low[index]])
                    else:
                        record[index] = '%d,%d' % (QI_ORDER[index][partition.low[index]],
                                                   QI_ORDER[index][partition.high[index]])
                elif isinstance(record[index], str):
                    if partition.low[index] == partition.high[index]:
                        record[index] = QI_ORDER[index][partition.low[index]]
                    else:
                        record[index] = QI_ORDER[index][partition.low[index]] + \
                                        ',' + QI_ORDER[index][partition.high[index]]
            result.append(record)
    # If you want to get NCP values instead of percentage
    # please remove next three lines
    ncp /= QI_LEN
    ncp /= data_size
    ncp *= 100
    # ncp /= 10000
    if __DEBUG:
        from decimal import Decimal

    return (result, (ncp, rtime))


def scan_ontology_folder(folder_path):
    pathDir = os.listdir(folder_path)
    ontology_dict = {}

    for file_name in pathDir:
        file_path = folder_path + '/' + file_name
        ontology_dict[file_name.split('.')[0]] = file_path

    return ontology_dict


def get_fd_columns(fd_list):
    """
    
    :param fd_list: a list of fd object
    :return: 把所有设计了fd的column放到一个list 中 
    """
    columns = []

    for fd in fd_list:
        columns = columns + fd.rhs + fd.lhs

    columns = list(set(columns))

    return columns


class OntologyNode():
    """
    the level of root node is 0.

    """

    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None

    def get_value(self):
        return self.value

    def set_value(self, v):
        self.value = v

    def is_leaf(self):
        tag = False
        if len(self.children) == 0:
            tag = True
        return tag

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        """
        get the list of children (node)
        """
        return self.children

    def get_leafnodes(self):
        """
        return a list of leaf values
        """

        result = []

        if self.is_leaf():
            result.append(self.value)

        else:
            children = self.get_children()
            for child in children:
                result.append(child.get_leafnodes())
        return list(flatten(result))  ###return values

    def get_level(self):
        """
        get the level of current node. Root node is 0.
        """
        if self.parent is None:
            return 0
        else:
            return self.parent.get_level() + 1

    def get_root_value(self):
        if self.parent:
            return self.value
        else:
            up = self.get_parent()
            return up.get_root_value()

    def __str__(self):
        r = ['value: ', self.value, 'children: ', self.children]
        return ' '.join(r)


def value_to_node(root, value):
    """
    input value, return the corresponding node object, otherwise return none

    Args:
        root: the root node of the tree
        value: the value of the node we are looking for
    Return:
        the node object or none.
    """
    if root.value == value:
        return root
    else:
        childrenList = root.get_children()
        for child in childrenList:
            tmp = value_to_node(child, value)
            if tmp is not None:
                return tmp


def get_ancestors(root, node):
    """
    given the node, return its all ancestors
    传入node， 得到所有祖先的value list

    Args:
        root
        node
    Return:
        a list of ancestor values of this node
    """
    ancestorList = []

    def get_ancestor_help(root, node, ancestorList):
        if node.get_parent() is None:
            return ancestorList
        else:
            currentParent = node.get_parent()
            ancestorList.append(currentParent.get_value())
            get_ancestor_help(root, currentParent, ancestorList)
            return ancestorList

    return get_ancestor_help(root, node, ancestorList)


def get_all_leafnodes(root, node):
    """
    given the node, return its all leaf node (in the list of value form)
    如果输入的node是 leaf, 那么就直接将该node 输出
    """
    childrenList = []

    # help function to maintain the list
    def get_all_leaf_help(root, node, childrenList):
        if node.is_leaf():
            childrenList.append(node.get_value())
            return childrenList
        else:
            for child in node.get_children():
                #                 childrenList.append(child.getValue())
                get_all_leaf_help(root, child, childrenList)
            return childrenList

    return list(flatten(get_all_leaf_help(root, node, childrenList)))


def get_all_descendants(root, node):
    """
    given the node, return its all children (not only leaf node, but all children till leaf)
    如果输入的 node 是leaf, 那么输出就是 none
    """
    childrenList = []

    def get_all_children_help(root, node, childrenList):
        """
        如果没有孩子, 就不做操作;
        如果有孩子, 就把孩子加进去, 再递归
        """
        if node.get_children() is None:
            return childrenList
        else:
            for child in node.get_children():
                childrenList.append(child.get_value())
                get_all_children_help(root, child, childrenList)
            return childrenList

    return get_all_children_help(root, node, childrenList)


def get_node_by_level(root, node, level):
    """
    given the root, node, and level, return the node of the specified level
    if the level is out of the ground node level, then it returns leaf node
    """
    current_level = node.get_level()
    # root is level 0
    level_difference = current_level - level
    for i in range(level_difference):
        node = node.get_parent()
    return node


def flatten(items):
    """Yield items from any nested iterable;
    >>> l = [1, 2, 3, [3, 4, 5, [7, 8]]]
    >>> f = list(flatten(l))
     [1, 2, 3, 3, 4, 5, 7, 8]
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


class fd():
    """
    fd object, lhs指左边, rhs指右边 (存的column index)
    lhs_by_name, 是 value 形式的 (column name value)
    """

    def __init__(self):
        self.lhs = []
        self.rhs = []
        self.lhs_by_name = []
        self.rhs_by_name = []


class ErrorCell():
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.error_value = value
        self.involved_errors = 0  ##分子
        self.total_errors = 0  ##分母
        self.error_factor = 0  ##求值
        self.budget = 0
        self.price = 0
        self.repair_values = None
        self.final_repair = None

    def __str__(self):
        c = ['row: ', self.row, 'col', self.col, 'value', self.error_value, 'repair_value', self.repair_values,
             'final_repair', self.final_repair]
        return ''.join(c)


def read_fd(fd_path):
    """
    读取fd, 你之前的方法要重新写
    fd文件的形式, 第一行, 所有的column;
    第二行: drag,sym|illness
    通过"|" 分割
    :param fd_path:输入fd的文件位置
    :return: a list of fd 对象
    """
    file = open(fd_path)
    fd_list = []
    line = 0
    columns = []
    for row in file:
        if line == 0:
            cols = row.split(',')
            for col in columns:
                col.strip()
                columns.append(col)
        else:
            new_fd = fd()

            str_fd = row.split('|')
            left_side = str_fd[0]
            right_side = str_fd[1]
            left_columns = left_side.split(',')
            right_columns = right_side.split(',')
            for column in left_columns:
                new_fd.lhs_by_name.append(column.strip())
            for column in right_columns:
                new_fd.rhs_by_name.append(column.strip())

            fd_list.append(new_fd)
        line = line + 1

    return fd_list


def calculate_cell_err_per_fd(fd, df_target, violate_cell):
    """
    计算在给定一个fd下, cell涉及的error 个数, 这个方法是在计算error_factor的时候用到的
    给定一个error_cell, 计算所有的 violation
    """
    left_cols_fd = fd.lhs
    right_cols_fd = fd.rhs
    left_pattern, right_pattern = [], []

    cols = fd.lhs + fd.rhs

    if violate_cell.col not in cols:
        return 0

    for col in left_cols_fd:
        left_pattern.append(df_target.iat[violate_cell.row, col])
    for col in right_cols_fd:
        right_pattern.append(df_target.iat[violate_cell.row, col])

    error = 0

    for row_index in range(0, len(df_target)):
        check_left_pattern, check_right_pattern = [], []
        for col in left_cols_fd:
            check_left_pattern.append(df_target.iat[row_index, col])
        if check_left_pattern == left_pattern:
            for col in right_cols_fd:
                check_right_pattern.append(df_target.iat[row_index, col])
            if check_right_pattern != right_pattern:
                error = error + 1

    if error > 0:
        return error + 1
    else:
        return error


def calculate_cell_err(fd_list, df_target, violate_cell):
    """
    计算error factor 的分子
    Args:
        fd_path: fd路径
        df_target: 只还有ground value, 没有general value
        cell_location: cell的坐标, 比如(1,2), 也可以是一个cell对象, 然后
        再取坐标
    Return:
        当前cell 涉及到的error的个数, 也就是公式的分子
    """
    # 根据cell_location的column, 来确定这个cell涉及了哪几个fd
    errors = 0
    for fd in fd_list:
        current_errors = calculate_cell_err_per_fd(fd, df_target, violate_cell)
        errors = errors + current_errors
    return errors


def find_violations(df, cols, length_right_side=1):
    """
    计算的思想是:
    1. 根据整个FD (X,Y), 用 groupby(FD)找出所有的pattern;
    2. 这些pattern就是 .groups 得到的key, 对应的row index就是 value
    3. 针对X, 也就是除去 .groups 的最后一个元素key[:-1], Counter 计算 X对应的所有pattern的frequency
    4. 对于每个 (X,Y) 的 pattern (k1,k2,k3), 如果发现 X 部分 (k1,k2)的frequency > 1, 就是violations

    E.g. : 如果对(X,Y)group后, 有4个pattern (1,2,2), (1,2,3), (1,4,5), (1,6,5);
    但是count这些pattern的X部分, 发现 (1,2)的frequency>1, 说明(1,2)出现了两次,
    说明4个pattern中, X 部分有相同的, 这些X相同的pattern就是violation

    Args:
        df: dataframe
        cols: 指定的columns, 是一个list, 如果 FD: (key1, key2) -> key3,
        那么就要传入cols = ['key1', 'key2', 'key3'].
    Return:
        a list of tuple, 每个tuple 由XY的pattern和对应的row index组成, 比如
        [((1, 2, 4), [1, 4]), ((1, 2, 3), [0])], 表示有两个violated pattern,
        pattern1 是 (1,2,4), 对应的row index是 [1,4]; row index从0 开始
    """
    violations = []
    cols_groups = df.groupby(cols)
    # groups 返回 dict (key = group的key, value = row index)
    groups_keys = cols_groups.groups.keys()
    from collections import Counter
    # t[:-1] 是 key1, key2, 没有key3, 也就是FD 的 X
    c = Counter([t[:-length_right_side] for t in groups_keys])
    from collections import namedtuple
    # 使用的时候, 要用 V(group, index) 初始化, 而不是 用 Violation名字, 这个名字只是内部标识
    V = namedtuple('Violation', ['pattern', 'rowIndex'])
    for group, index in cols_groups.groups.items():
        # 对于k1k2 ->k3 来说, 如果k1,k2,k3的所有group中
        # k1,k2 对应的group多过 1, 说明 k3 有不一样的
        if c.get(group[:-length_right_side]) > 1:
            violation = V(group, index)
            violations.append(violation)
    return violations


def calculate_error_per_fd(fd, df_target):
    """
        一个fd下的所有 violation的 个数
        这个方法是多余的, 重复调用了find_violations, 可以和其他方法整合
        Args:
            fd, fd 对象
        Return:
            一个fd下的所有 violation的row 的个数
    """
    cols = fd.lhs + fd.rhs
    cols_name = []
    for c in cols:
        cols_name.append(df_target.columns[c])
    length_right_side = len(fd.rhs)
    num_violation = 0
    Violation = find_violations(df_target, cols_name, length_right_side)
    # 根据我的方法, 返回的violation中含有pattern, 每个pattern下有row
    for pattern in Violation:
        num_violation = num_violation + len(pattern.rowIndex)
    # 返回当前fd下, 所有violation的row 的个数
    return num_violation


def calculate_total_errors(fd_list, df_target):
    """
        计算当前 target在所有 fd下的 violation 个数,
        也就是公式分母
    """
    total_errors = 0
    for fd in fd_list:
        num_violation = calculate_error_per_fd(fd, df_target)
        total_errors = num_violation + total_errors
    return total_errors


### 上面都是在 target 只有 ground value下进行计算的

def find_violate_cells(fd_list, df_target, rhs_only=False):
    '''
    给定 a list of fd 对象, 和target, 找到所有的violate_cell 也就是参与了violation的cell
    把这些cell的位置信息以及当前的error value, 组成一个ErrorCell obj, 放入一个list中返回
    一个violation, 会把当前fd下的所有cell都放入(因为不知道到底哪个cell是错误的, 所有全部放入),
    因此返回的error cell的个数是远大于 violation 的个数的
    :param fd_list:a list of fd object
    :param df_target:
    :return: 返回violate_cell的list
    '''
    violate_cells = []
    position_tuples_of_violate_cells = []
    for fd in fd_list:
        # 找出当前fd 下的所有 参与了violation的cell 的 位置 a list of [row,col]
        violation_cell_position_list = find_violate_cell_position_per_fd(fd, df_target, rhs_only)
        for v in violation_cell_position_list:
            if v not in position_tuples_of_violate_cells:
                position_tuples_of_violate_cells.append(v)
    for t in position_tuples_of_violate_cells:
        value = df_target.iat[t[0], t[1]]
        violate_cell = ErrorCell(t[0], t[1], value)
        violate_cells.append(violate_cell)

    return violate_cells


def find_violate_cell_position_per_fd(fd, df_target, rhs_only=False):
    '''
    输入单个fd和target, 找出所有violation是position (row, col), 
    然后返回 a list of position
    :param fd: a fd object
    :param df_target:
    :return: a list of position in the form of [row, col]   position tuple: [row,col]
    '''
    cols = fd.lhs + fd.rhs
    cols_name = []
    for c in cols:
        cols_name.append(df_target.columns[c])
    length_right_side = len(fd.rhs)
    # 前面的计算只是为了调用我写的find_violaitons 方法
    # 该方法返回 a list of Violations, 这个violation是一个namedtupe, 由(pattern, rowindex)组成
    violation_list = find_violations(df_target, cols_name, length_right_side)
    violate_rows = []

    for v in violation_list:
        for row in v.rowIndex:
            violate_rows.append(row)

    position_tuples = []
    # 如果假设error都在右边, 那么就只用把右边的col添加进去, 而不用把所有的都添加进去
    if rhs_only:
        for row in violate_rows:
            position_tuples.append([row, fd.rhs[0]])
    else:
        for row in violate_rows:
            for col in cols:
                if [row, col] not in position_tuples:
                    position_tuples.append([row, col])

    return position_tuples


def calculate_error_facotors(fd_list, df_target, rhs_only=False):
    '''
    error_factor 的计算是 当前cell value 参与了的violation 的个数/所有的violation的个数
    这个方法返回的是 a list of error cell, 是 cell的个数    
    :param fd_list: a list of fd object
    :param df_target:
    :return:  list of violate_cells ,每个cell的错误属性值都设置好了
    '''
    # 找到所有参与了violation的cell, 返回 a list of ErroCell, 包含位置信息和err value
    # 因为一个violation下所有的cell都会被放入, 因此error cell 个数远大于 violation 个数
    # 这里可以在里面优化, 如果知道error 都在rhs的话, 那么就只会放入rhs的cell, 那样
    # error cell的个数就和 vioaltion 个数接近了
    violate_cells = find_violate_cells(fd_list, df_target, rhs_only)

    # 进行optimization, 不要这样计算total violation
    # 把原本的calculate_total_error 去掉了, 用已经生成的violate_cells来计算total_error
    # t_err = calculate_total_errors(fd_list, df_target)
    total_violation_row_index = []
    for c in violate_cells:
        total_violation_row_index.append(c.row)
    # 所有fd下, 总的violation的个数
    total_violation = len(set(total_violation_row_index))

    for cell in violate_cells:
        # 计算当前cell参与了的violationd 的个数
        invol_error = calculate_cell_err(fd_list, df_target, cell)
        err_fac = invol_error / total_violation
        cell.involved_errors = invol_error
        cell.total_errors = total_violation
        cell.error_factor = err_fac

    return violate_cells


def divide_budget(budget, violate_cells):
    ref_vio_cell = violate_cells

    for c in ref_vio_cell:
        c.budget = c.error_factor * budget

    return ref_vio_cell


def find_repair_value(violate_cells, df_master, df_target, threshold):
    '''
    给 violate_cell 添加了 a list of repair values
    :param violate_cells:
    :param df_master:
    :return:
    '''

    ref_vio_cell = violate_cells
    for cell in ref_vio_cell:
        matched_df = find_matched_df_ground(cell, df_target, df_master, threshold)

        cell.repair_values = matched_df[matched_df.columns[cell.col]].unique()

    return ref_vio_cell


def rank_cell(violate_cells, reverse=True):
    """
    根据 error factor 来排序, 从大到小.
    :param violate_cells:
    :param reverse:
    :return:
    """

    def cmp(cell1, cell2):
        return cell1.error_factor - cell2.error_factor

    def cmp_to_key(cmp):
        'Convert a cmp= function into a key= function'

        class K(object):
            def __init__(self, obj, *args):
                self.obj = obj

            def __lt__(self, other):
                return cmp(self.obj, other.obj) < 0

            def __gt__(self, other):
                return cmp(self.obj, other.obj) > 0

            def __eq__(self, other):
                return cmp(self.obj, other.obj) == 0

            def __le__(self, other):
                return cmp(self.obj, other.obj) <= 0

            def __ge__(self, other):
                return cmp(self.obj, other.obj) >= 0

            def __ne__(self, other):
                return cmp(self.obj, other.obj) != 0

        return K

    ref_violated_cells = violate_cells
    ref_violated_cells = sorted(ref_violated_cells, key=cmp_to_key(cmp), reverse=reverse)
    return ref_violated_cells


def set_price_for_violate_cells(violate_cells, pricing_df, price_factor):
    ref_violate_cells = violate_cells

    for cell in ref_violate_cells:
        row = cell.row
        col = cell.col
        price = pricing_df.ix[row, col] * price_factor
        cell.price = price

    return ref_violate_cells


def do_repair(df_master, pricing_df, violate_cells, budget, k_anonymity):
    '''
    剩下的budget是否大于0》》cell的budget是否大于price》》len（repair_values）是否大于k》》do repair

    :param df_master:
    :param pricing_df:
    :param violate_cells:
    :param budget:
    :param k_anonymity:
    :return:
    '''

    total_budget = budget

    repaired_cell_list = []

    for cell in violate_cells:

        print('left budget:' + str(total_budget))

        if total_budget > 0:

            if cell.budget > cell.price:

                print('budget: ' + str(cell.budget), 'price: ' + str(cell.price))
                print('cell.repair_values: ' + str(cell.repair_values))
                print('length of repair values: ' + str(len(cell.repair_values)))
                print('k: ' + str(k_anonymity))

                if len(cell.repair_values) >= k_anonymity:

                    print('repair values length: ' + str(len(cell.repair_values)), 'k: ' + str(k_anonymity))

                    rsl = do_repair_per_cell(cell, df_master=df_master, pricing_df=pricing_df)

                    print(rsl)

                    if rsl != False:
                        repaired_cell_list.append(rsl)

                    total_budget = total_budget - cell.price

        else:

            break

    return repaired_cell_list


def query_bundle(df_master, pricing_df, violate_cells, budget, k_anonymity, ontology_dict):
    total_budget = budget

    while budget > 0:

        level = 0

        for cell in violate_cells:

            if budget > 0:

                col = cell.col
                ontology_node = value_to_node(cell.value, ontology_dict[col])
                query_rsl = query_find_value(level=level, pricing_df=pricing_df)
                value = query_rsl[0]
                price = query_rsl[1]
                if price <= budget:
                    budget = budget - price
                    cell.repair_values = value

                else:
                    break

                level = level + 1

            else:

                break


def evaluation(repaired_cell_list, df_master, error_num):
    num_total_repair = len(repaired_cell_list)

    num_correct_repair = 0

    for cell in repaired_cell_list:

        row = cell.row
        col = cell.col

        correct_value = df_master.ix[row, col]

        if cell.final_repair == correct_value:
            num_correct_repair = num_correct_repair + 1

    print('num_correct_repair: ' + str(num_correct_repair))

    print('num_total_repair: ' + str(num_total_repair))

    precision = 0

    recall = 0

    if num_total_repair == 0:

        precision = 0

    else:

        precision = num_correct_repair / num_total_repair

    if error_num == 0:

        recall = 0

    else:

        recall = num_correct_repair / error_num

    return precision, recall


def do_repair_per_cell(violate_cell, df_master, pricing_df):
    '''
    把match 到的repair_values list 里的value，到master里定位，看价格，然后从小到大排序，选第一个最便宜的作为final_repair

    :param violate_cell:
    :param df_master:
    :param pricing_df:
    :return:
    '''

    pending = violate_cell.repair_values

    col = violate_cell.col

    price_dic = {}

    for value in pending:

        if value is not '':

            for row in range(0, len(df_master)):

                if df_master.ix[row, col] == value:
                    print(df_master.ix[row, col])

                    price_dic[value] = pricing_df.ix[row, col]

                    print(price_dic[value])

                    break

    print(price_dic)

    if len(price_dic) > 0:

        price_dic = sorted(price_dic.items(), key=lambda x: x[1], reverse=False)

        violate_cell.final_repair = price_dic[0][0]

        print(violate_cell.final_repair)

        return violate_cell

    else:

        return False


def calculate_price(sensitive_weight, distinct_weight):
    '''

    :param sensitive_weight:
    :param distinct_weight:
    :return:
    '''

    price = distinct_weight * sensitive_weight

    return price


def calculate_column_weight(col_weight_dict):
    """
    计算每个column的weight. 把给定的weight进行normalize
    :param col_weight_dict: 传入一个dict, key为column_index, value 为人为设置的重要程度
    :return: 返回一个dict, key为col_index, value 为 normalize后的weight
    """
    ref_col_weight_dict = col_weight_dict
    total = 0
    for item in ref_col_weight_dict:
        total = total + ref_col_weight_dict[item]
    for item in ref_col_weight_dict:
        ref_col_weight_dict[item] = ref_col_weight_dict[item] / total
    return ref_col_weight_dict


def generate_pricing_df_per_col(df_master, column, column_weight):
    '''

    :param df_master:
    :param column_name:
    :param column_weight:
    :return:
    '''

    col = column

    col_df = copy.deepcopy(df_master[df_master.columns[col]])

    distinct_values_dict = create_distinct_values_dict(df_master, column)

    distinct_size = len(df_master)

    cell_price_tmp_list = []

    cell_price_saving_list = []

    total_price = 0

    for row in range(0, len(col_df)):
        value = col_df.ix[row, 0]

        frequency = distinct_values_dict[value]

        distinct_weight = distinct_size / frequency

        price = calculate_price(column_weight, distinct_weight)

        total_price = total_price + price

        cell_price_tmp_list.append(price)

    for price in cell_price_tmp_list:
        final_price = (price / total_price) * column_weight

        cell_price_saving_list.append(final_price)

    return cell_price_saving_list


def generate_pricing_df(df_master, columns_weight_dic):
    """
    计算 master 的price

    :param df_master:
    :param columns_weight_dic: normalize 后的column_weight dict
    :return: 一个 price df
    """
    pricing_df = copy.deepcopy(df_master)

    for col in range(0, len(pricing_df.columns)):

        pricing_df_col = generate_pricing_df_per_col(df_master=pricing_df, column=col,
                                                     column_weight=columns_weight_dic[col])

        for row in range(0, len(pricing_df)):
            pricing_df.ix[row, col] = pricing_df_col[row]

    return pricing_df


def create_distinct_values_dict(df_master, column_index):
    '''

    :param df_master:
    :param column_name:
    :return: a dict, key: distinct_value, value: it's frequency
    '''

    col = column_index

    distinct_value_dict = {}

    for row in range(0, len(df_master)):

        value = df_master.ix[row, col]

        if value in distinct_value_dict:

            distinct_value_dict[value] = distinct_value_dict[value] + 1

        else:

            distinct_value_dict[value] = 1

    return distinct_value_dict


class UpdateCell():
    def __init__(self, row, column, old_value):
        self.row = row
        self.column = column
        self.old_value = old_value
        self.update_value = None

    def __str__(self):
        c = ['row: ', str(self.row), ' column: ', str(self.column), ' old_value: ', str(self.old_value),
             ' update_value: ', str(self.update_value)]
        return ''.join(c)


def generate_target_old(df_master, fd_list, error_num, percentage):
    '''
    随机生成一些cell的坐标, 把他们的value 换成该列其他的随机值,同时要保留以前的 value 信息,
    作为 ground truth. percentage 控制master的前 百分之多少row作为 target. 可以用到之前的update_cell中的方法
    :param df_master:
    :param error_num:
    :param percentage:
    :return: 生成 df_target, 和 被修改过的cell list
    '''

    master = copy.deepcopy(df_master)
    fdcolumns = get_fd_columns(fd_list)
    df_size = len(master)
    target_size = int(df_size * percentage)
    update_df = master.iloc[:target_size]
    updatecells_list = []

    for num in range(0, error_num):
        random_col = random.choice(fdcolumns)
        random_row = random.randint(0, target_size - 2)
        value = update_df.iat[random_row, random_col]
        uc = UpdateCell(random_row, random_col, value)
        updaterow = random.randint(0, len(master) - 1)
        update_df.iat[random_row, random_col] = master.iat[updaterow, random_col]
        uc.update_value = master.iat[updaterow, random_col]
        updatecells_list.append(uc)

    return update_df, updatecells_list


def generate_target(df_master, fd_list, error_num, percentage):
    """
    随机生成一些cell的坐标, 把他们的value 换成该列其他的随机值,同时要保留以前的 value 信息,
    作为 ground truth. percentage 控制master的前 百分之多少row作为 target. 可以用到之前的update_cell中的方法
    :param df_master:
    :param error_num:
    :param percentage:
    :return: 生成 df_target, 和 被修改过的cell list
    """
    # 生成原始的target
    df_target_size = int(len(df_master) * percentage)
    df_target = copy.deepcopy(df_master.iloc[:df_target_size])
    update_cell_list = []

    # fd_columns = get_fd_columns(fd_list) 这个是把error放到所有的fd_column下
    fd_columns = [e for fd in fd_list for e in fd.rhs]  # 这个是放到右边

    # 如果只把错误放在rhs
    for i in range(error_num):
        random_col = random.choice(fd_columns)
        random_row = random.randint(0, df_target_size - 1)
        old_value = df_target.iat[random_row, random_col]
        new_cell = UpdateCell(random_row, random_col, old_value=old_value)
        # 为了让new_value 和 old_value 不重复
        col_unique_value = list(df_master.iloc[:, random_col].unique())
        col_unique_value.remove(old_value)
        # 从去除了old_value的list中 随机
        new_value = random.choice(col_unique_value)
        new_cell.update_value = new_value
        df_target.iat[random_row, random_col] = new_value
        update_cell_list.append(new_cell)
    return df_target, update_cell_list


def load_master(data_path, fd_path):
    '''
    加载masterdata， 只保留fd columns
    FD 对象也要改！！！！！！！！！！！！
    :param data_path:
    :param fd_path:
    :return:
    '''
    fd_list = read_fd(fd_path)
    df_master = pd.read_csv(data_path, header=0, encoding="ISO-8859-1")
    index_columns = df_master.columns
    columns = []
    for c in index_columns:
        columns.append(c)

    for fd in fd_list:
        left_columns = fd.lhs_by_name
        for column in left_columns:
            fd.lhs.append(columns.index(column))
        right_columns = fd.rhs_by_name
        for column in right_columns:
            fd.rhs.append(columns.index(column))
    return df_master, fd_list


def set_fd_column_by_name_per_fd(df, fd):
    '''

    :param df:
    :param fd:
    :return:
    '''

    columns = df.columns
    update_fd = fd
    for col in fd.lhs:
        update_fd.lhs_by_name.append(columns[col])
    for col in fd.rhs:
        update_fd.rhs_by_name.append(columns[col])
    return update_fd


def set_fd_column_by_name(df, fd_list):
    '''
    对df进行删列操作前进行，为每个fd对象更新 rhs_by_name 和 lhs_by_name属性
    :param df_master:
    :param fd_list:
    :return: 返回新的fd_list
    '''

    ref_fd_list = fd_list

    updated_fd_list = []

    for fd in ref_fd_list:
        fd = set_fd_column_by_name_per_fd(df, fd)

        updated_fd_list.append(fd)

    return updated_fd_list


def update_fd_column_by_index_per_fd(df_master, fd):
    '''
    删掉非fd列后，更新fd对象的列index位置
    :param df_master:
    :param fd:
    :return:
    '''
    update_fd = fd
    cols = df_master.columns
    columns = []
    for col in cols:
        columns.append(col)
    fd.rhs = []
    fd.lhs = []
    for col in fd.lhs_by_name:
        idx = columns.index(col)
        fd.lhs.append(idx)
    for col in fd.rhs_by_name:
        idx = columns.index(col)
        fd.rhs.append(idx)
    return update_fd


def update_fd_column_by_index(df_master, fd_list):
    '''
    改变fd 列的相对位置
    :param df_master:
    :param fd_list:
    :return:
    '''

    update_fd_list = []

    for fd in fd_list:
        update_fd = update_fd_column_by_index_per_fd(df_master, fd)
        update_fd_list.append(update_fd)

    return update_fd_list


def check_df_per_fd(df_master, fd):
    '''
    检查数据，是否符合fd , 对于不符合fd的小部分数据, 进行修复. 把它们改为 第一个 出现的row的value (TODO,frequency 最高的 value).
    :param df_master:
    :param fd:
    :return:
    '''
    ref_df_master = df_master
    lhs = fd.lhs
    rhs = fd.rhs
    check_dict = {}
    for row in range(0, len(df_master)):
        left_pat = ''
        right_pat = ''
        # 这里lhs是fd 的column index
        for col in lhs:
            left_pat = left_pat + '*' + ref_df_master.ix[row, col]
        for col in rhs:
            right_pat = right_pat + '*' + ref_df_master.ix[row, col]
        if left_pat not in check_dict:
            check_dict[left_pat] = right_pat
        else:
            if right_pat != check_dict[left_pat]:

                update_pat = check_dict[left_pat].split('*')
                correct_pat = update_pat[1:]

                for c in range(0, len(rhs)):
                    ref_df_master.ix[row, rhs[c]] = correct_pat[c]

    return ref_df_master


def check_df(df_master, fd_list):
    '''
    检查数据，是否符合fd, 对于不符合fd的小部分数据, 进行修复. 把它们改为 frequency 最高的 value.
    :param df_master:
    :param fd_path:
    :return:
    '''

    ref_df_master = copy.deepcopy(df_master)

    for fd in fd_list:
        ref_df_master = check_df_per_fd(ref_df_master, fd)

    return ref_df_master

def get_col_dict(df):
    """
    优化 find_match_rows的辅助方法
    直接返回 a list of dict, key是df中的original value, value为他们的position
    match的时候, 可以直接通过这些dict来进行定位, 直接来找position
    Args:
        df
    Returns:
        a list of dict, 每个包含来当前column的所有value, 以及他们的pisition
    """
    dict_list = []
    for col in df.columns:
        col_dict = col_to_dict(df[col])
        dict_list.append(col_dict)
    return dict_list

def col_to_dict(serie):
    """
    优化 find_match_rows的辅助方法

    把一个serie的value全部换成一个dict, key是original_value, value为 a list of position.
    比如, [a,a,b,c,c] 就被换成 {a:[0,1], b:[2], c:[3,4]}
    """
    final_dict = {}
    for i,v in serie.iteritems():
        if v not in final_dict:
            row_index_list = []
            # 这个地方必须这样分开写, 不能 final_dict[v] = row_index_list.append(i)
            row_index_list.append(i)
            final_dict[v] = row_index_list
        else:
            final_dict[v].append(i)
    return final_dict


def find_matched_rows(serie_target, threshold, col_dict_list):
    """
    Args:
        serie_target: 一个 target的 row 对象
        col_dict_list: a list of col_dict, 从get_col_dict得到
    Return:
        a list of row index, 满足matching threshold
    """
    columns_size = len(serie_target)
    minimal_true = int(columns_size * threshold)
    matched_row_index = []

    total_row_index = []
    for i in range(len(r)):
        col_dict = col_dict_list[i]
        # the row index of current column that contains the attribute value, row_index is a list of int
        row_index = col_dict[r[i]]
        total_row_index = total_row_index + row_index
    from collections import Counter
    c = Counter(total_row_index)
    for index, freq in c.items():
        if freq >= minimal_true:
            matched_row_index.append(index)
    return matched_row_index


def find_matching_rows(violate_cell, record, df_master, threshold, fd_list):
    """
    TODO: 优化： 数到了minimal_true 停
                肯定达不到minimal_true 停
    given a record from target, find out all the matched record records in master.
    return the row index in the df_master
    :param record:
    :param df_master:
    :return: a list of matched records
    """
    row_index = []
    columns = df_master.columns
    fd_column_size = len(get_fd_columns(fd_list))
    # TODO: check the minimal_true size
    columns_size = len(columns) - fd_column_size
    row_dic = {}
    minimal_true = int(columns_size * threshold)

    for i in range(0, len(df_master)):
        row_dic[i] = 0

    for i in range(0, len(columns)):
        if i != violate_cell.col:
            # 这个是用在general value的时候
            # column_check = df_master[columns[i]].isin(record[i])
            column_check = df_master[columns[i]] == record[i][0]  # 如果只有ground value, 就只有取第一个, 因为只有一个value
            for k in range(0, len(column_check)):
                if column_check[k] == True:
                    row_dic[k] = row_dic[k] + 1

    for row in row_dic:
        if minimal_true == 0:
            minimal_true = 1
        if row_dic[row] >= minimal_true:
            row_index.append(row)
    return row_index


def matched_df_generator(df_master, rows_index):
    matched_df = df_master.iloc[rows_index]
    return matched_df


def find_matches(violate_GeValue, df_master, geColumn, threshold):
    """
    把需要match的cell所在的row，除了cell所在列，其他所有的列的数据已list的形式存入一个list。如果是general value，存本身以及它之下所有的ground value。
    check的时候只需查看对应列的数据在不在list里面
    The matching should be defined as semantic matching. Record may contain general values, master only
    contain ground values. If the ground value is one of children in the general value, then it will be considered
    as matched.
    返回的是符合general matching 的 row index

    :param row: Series object, which can be got from df_target.ix[row_id]
    :param df_master:
    :param geColumn: dictionary, key is column index, and value is the root node object.
    :threshold: matching number in a row
    :return: a list of matched row index
    """

    data = df_master.ix[violate_GeValue.row]
    print(data)
    record = []
    print(len(data))

    for i in range(0, len(data)):
        if i in geColumn:
            root = geColumn[i]
            value = data[i]
            node = value_to_node(root, value)
            value_set = get_all_leafnodes(root, node)
            value_set.append(value)
            record.append(value_set)
        else:
            record.append([data[i]])

    return find_matching_rows(violate_GeValue, record, df_master, threshold)


def find_matched_df_ground(violate_cell, df_target, df_master, threshold):
    """
    把需要match的cell所在的row，除了cell所在列，其他所有的列的数据已list的形式存入一个list。如果是general value，
    存本身以及它之下所有的ground value。
    check的时候只需查看对应列的数据在不在list里面
    :param row: Series object, which can be got from df_target.ix[row_id]
    :param df_master:
    :param geColumn: dictionary, key is column index, and value is the root node object.
    :threshold: matching number in a row
    :return: a list of matched row index
    """
    data = df_target.ix[violate_cell.row]
    record = []

    for i in range(0, len(data)):
        record.append([data[i]])

    row_index = find_matching_rows(violate_cell, record, df_master, threshold)
    matched_df = matched_df_generator(df_master, row_index)

    return matched_df


def budget_price_filter(divided_budget_violate_cells):
    '''
    过滤掉budget<price的cell
    :param divided_budget_violate_cells:
    :return:
    '''

    violate_cells = []

    for cell in divided_budget_violate_cells:
        if cell.budget > cell.price:
            violate_cells.append(cell)
    return violate_cells


def query_find_value(level, pricing_df):
    values = []

    return values


def time_performance_by_num_of_records(master_data_list, fd_path):
    file_object = open('time_performance_by_num_of_records.txt', 'w')
    for data in master_data_list:
        rsl = integrity_experiment(data, fd_path)
        file_object.writelines(str(rsl[0]) + '\n')
        file_object.writelines(str(rsl[1]) + '\n')
        file_object.writelines('-----' + '\n')
    file_object.close()


def set_up_impact_budget_accuracy(master_data_path, fd_path, target_percentage=0.3, e=0.05, total_price=1000):
    """

    :param master_data_path:
    :param fd_path:
    :param target_percentage:
    :param e: error rate, error 在target中的比例
    :param total_price:
    :return:
    """
    load = load_master(master_data_path, fd_path)
    df_master = load[0]
    df_master = df_master.iloc[:1000]
    fd_list = load[1]
    df_master = check_df(df_master, fd_list)
    master_size = len(df_master)
    error_rate = e
    error_num = int(master_size * target_percentage * error_rate)
    percentage = target_percentage
    PRICING_FACTOR = total_price
    generation = generate_target(df_master, fd_list, error_num, percentage)
    df_target = generation[0]
    # 设置每个column的weight, 主要用在计算price上面
    columns_weight_dic = {}
    for i in range(df_master.shape[1]):
        columns_weight_dic[i] = 1
    columns_weight_dic[22] = 2
    columns_weight_dic[26] = 3
    # normalize
    weights = calculate_column_weight(columns_weight_dic)
    # 生成 price
    pricing_df = generate_pricing_df(df_master, weights)
    # print(pricing_df)


    violate_cells = calculate_error_facotors(fd_list, df_target, True)

    # 这个是所有参与了violation的row的 在fd下的cell 的个数, 因此比较大
    violate_cells_num = len(violate_cells)
    print("target size: " + str(len(df_target)))
    print("master size: " + str(len(df_master)))
    print("injected error number: " + str(error_num))
    print('violate_cells_num: ' + str(violate_cells_num))
    priced_violate_cells = set_price_for_violate_cells(violate_cells, pricing_df, price_factor=PRICING_FACTOR)
    print('violate_cells has been priced')
    ranked_violate_cells = rank_cell(priced_violate_cells)
    print('violate_cells has been ranked')

    return ranked_violate_cells, df_master, df_target, pricing_df, error_num, fd_list


def integrity_experiment(df_master, df_target, ranked_violate_cells, pricing_df, error_num, e, percentage, fd_list,
                         total_price=1000, B=0.1, threshold=0.9, k=1):
    """
    整合实验
    :param master_data_path:
    :param fd_path:
    :param e
    :param B:
    :param threshold:
    :param k:
    :return:
    """
    t0 = time.time()

    BUDGET = B * total_price * percentage
    THRESHOLD = threshold
    K = k

    divided_budget_violate_cells = divide_budget(budget=BUDGET, violate_cells=ranked_violate_cells)

    print('violate_cells has been divided budget')
    # for cell in divided_budget_violate_cells:
    #     print('price: ' + str(cell.price), 'budget' + str(cell.budget))
    violate_cells = budget_price_filter(divided_budget_violate_cells)
    found_repair_violate_cells = find_repair_value(violate_cells, df_master, df_target, threshold=THRESHOLD,
                                                   fd_list=fd_list)
    print(len(found_repair_violate_cells))
    print('repair values have been found')
    # match 到三个cell, 选一个最便宜的进行购买
    repaired_cell_list = do_repair(df_master, pricing_df, violate_cells, budget=BUDGET, k_anonymity=K)
    print('violate_cells has been repaired')
    t1 = time.time()
    for c in repaired_cell_list:
        print(c.final_repair)

    total_time = t1 - t0

    eval = evaluation(repaired_cell_list, df_master, error_num)

    print('-----------------EXPERIMENT RESULT-----------------')

    print('')

    print('Size of the Master Dataset: ' + str(len(df_master)))

    print('Size of the Target Dataset: ' + str(len(df_target)))

    print('Error Rate: ' + str(e))

    print('Budget: ' + str(BUDGET))

    print('Price: ' + str(total_price))

    # price 是对df——master所有cell进行price

    print('K_anonymity: ' + str(K))

    print('+++++++++++')

    print('time: ' + str(total_time))

    print('precision: ' + str(eval[0]))

    print('recall: ' + str(eval[1]))

    size_of_master_dateset = str(len(df_master))

    return (size_of_master_dateset, total_time, str(eval[0]), str(eval[1]))


def impact_budget_accuracy(master_data_path, fd_path, target_percentage=0.3, e=0.05, total_price=1000, threshold=0.9,
                           k=1):
    budget = 0.1

    filename = 'experiment_result/impact_budget_accuracy/budget.txt'

    file_object = open(filename, 'w')
    set_up = set_up_impact_budget_accuracy(master_data_path, fd_path, target_percentage, e, total_price)
    ranked_violate_cells = set_up[0]
    df_master = set_up[1]
    df_target = set_up[2]
    pricing_df = set_up[3]
    error_num = set_up[4]
    fd_list = set_up[5]
    while budget < 0.9:
        file_object.writelines('experiment ' + str(budget) + '\n')
        file_object.flush()
        rsl = integrity_experiment(df_master=df_master, df_target=df_target, ranked_violate_cells=ranked_violate_cells,
                                   pricing_df=pricing_df, error_num=error_num, e=e, percentage=target_percentage,
                                   fd_list=fd_list,
                                   total_price=1000, B=budget, threshold=threshold, k=k)
        file_object.writelines('budget: ' + str(budget) + '\n')
        file_object.writelines('excuting time: ' + str(rsl[1]) + '\n')
        file_object.writelines('precision: ' + str(rsl[2]) + '\n')
        file_object.writelines('recall: ' + str(rsl[3]) + '\n')
        file_object.writelines('-----' + '\n')
        file_object.flush()
        budget = budget + 0.1

    file_object.close()


if __name__ == '__main__':
    impact_budget_accuracy('./data_group/clinic_japan_trails/Japan_trials_original.csv',
                           './data_group/clinic_japan_trails/fds.csv', threshold=0.9)
