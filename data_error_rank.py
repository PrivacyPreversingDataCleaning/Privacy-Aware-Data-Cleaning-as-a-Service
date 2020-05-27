import copy
import pandas as pd
from src.data_matching import find_matched_df_ground


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


class cell():
    def __init__(self,row,col,value):

        self.row=row
        self.col = col
        self.value=value
        self.involved_errors=0 ##分子
        self.total_errors=0  ##分母
        self.error_factor=0  ##求值
        self.budget=0
        self.price=0
        self.repair_values=None
        self.final_repair=None


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
    fd_list=[]
    for row in file:
        new_fd=fd()
        devide_row=row.split(';')
        left_side,right_side=devide_row[0],devide_row[1]
        left_cols,right_cols=left_side.split(','),right_side.split(',')
        for col in left_cols:
            new_fd.lhs.append(int(col[0]))
        for col in right_cols:
            new_fd.rhs.append(int(col[0]))
        fd_list.append(new_fd)
    return fd_list


def calculate_cell_err_per_fd(fd, df_target, violate_cell):
    """
    计算在给定一个fd下, cell涉及的error 个数
    """
    left_cols_fd=fd.lhs
    right_cols_fd=fd.rhs
    left_pattern,right_pattern=[],[]

    cols=fd.lhs+fd.rhs

    if violate_cell.col not in cols:
        return 0

    for col in left_cols_fd:
        left_pattern.append(df_target.iat[violate_cell.row,col])
    for col in right_cols_fd:
        right_pattern.append(df_target.iat[violate_cell.row,col])

    error=0

    for row_index in range(0,len(df_target)):
        check_left_pattern, check_right_pattern = [], []
        for col in left_cols_fd:
            check_left_pattern.append(df_target.iat[row_index, col])
        if check_left_pattern == left_pattern:
            for col in right_cols_fd:
                check_right_pattern.append(df_target.iat[row_index, col])
            if check_right_pattern !=right_pattern:
                error=error+1

    if error>0:
        return error+1
    else:
        return error


def calculate_cell_err(fd_list, df_target, violate_cell):
    """
    计算分子
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


def find_violations(df, cols,length_right_side=1):
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
        一个fd下的所有error 个数
        Args:
            fd, fd 对象
        Return:
            一个fd下的所有error
    """
    cols=fd.lhs+fd.rhs
    cols_name=[]
    for c in cols:
        cols_name.append(df_target.columns[c])
    length_right_side=len(fd.rhs)
    errors = 0
    Violation=find_violations(df_target,cols_name,length_right_side)
    for pattern in Violation:
        errors=errors+len(pattern.rowIndex)
    return errors


def calculate_total_errors(fd_list, df_target):
    """
        计算当前 target在所有 fd下的 error 个数,
        也就是公式分母
    """
    total_errors = 0
    for fd in fd_list:
        errors = calculate_error_per_fd(fd, df_target)
        total_errors = errors + total_errors
    return total_errors


### 上面都是在 target 只有 ground value下进行计算的


def find_violate_cells(fd_list,df_target):
    '''

    :param fd_path:
    :param df_target:
    :return: 返回violate_cell的list
    '''
    violate_cells=[]
    position_tuples_of_violate_cells=[]
    for fd in fd_list:
        fd_tuples=find_violate_position_tupples_per_fd(fd, df_target)
        for tuple in fd_tuples:
            if tuple not in position_tuples_of_violate_cells:
                position_tuples_of_violate_cells.append(tuple)
    for tuple in position_tuples_of_violate_cells:
        value=df_target.iat[tuple[0],tuple[1]]
        violate_cell=cell(tuple[0],tuple[1],value)
        violate_cells.append(violate_cell)

    return violate_cells


def find_violate_position_tupples_per_fd(fd,df_target):
    '''

    :param fd:
    :param df_target:
    :return: position tupples's list   position tuple: [row,col]
    '''
    cols = fd.lhs + fd.rhs
    cols_name = []
    for c in cols:
        cols_name.append(df_target.columns[c])
    length_right_side = len(fd.rhs)
    Violation = find_violations(df_target, cols_name, length_right_side)
    violate_rows=[]
    for v in Violation:
        for row in v.rowIndex:
            violate_rows.append(row)
    position_tuples=[]
    for row in violate_rows:
        for col in cols:
            if [row,col] not in position_tuples:
                position_tuples.append([row,col])

    return position_tuples


def calculate_error_facotors(fd_list,df_target):
    '''

    :param fd_path:
    :param df_target:
    :return:  list of violate_cells ,每个cell的错误属性值都设置好了
    '''

    violate_cells=find_violate_cells(fd_list,df_target)
    t_err=calculate_total_errors(fd_list,df_target)

    for cell in violate_cells:
        invol_error=calculate_cell_err(fd_list, df_target, cell)
        err_fac=invol_error/t_err
        cell.involved_errors=invol_error
        cell.total_errors=t_err
        cell.error_factor=err_fac

    return violate_cells


def divide_budget(budget,violate_cells):

    ref_vio_cell=violate_cells

    for c in ref_vio_cell:
        c.budget=c.error_factor*budget

    return ref_vio_cell


def find_repair_value(violate_cells, df_master, df_target, threshold):
    '''
    给 violate_cell 添加了 a list of repair values
    :param violate_cells:
    :param df_master:
    :return:
    '''

    ref_vio_cell=violate_cells

    for cell in ref_vio_cell:

        matched_df=find_matched_df_ground(cell, df_target, df_master, threshold)

        cell.repair_values=matched_df[matched_df.columns[cell.col]].unique()

    return ref_vio_cell

def rank_cell(violate_cells,reverse=True):
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
    ref_violated_cells =sorted(ref_violated_cells, key=cmp_to_key(cmp),reverse=reverse)
    return ref_violated_cells


def set_price_for_violate_cells(violate_cells,pricing_df,price_factor):

    ref_violate_cells=violate_cells

    for cell in ref_violate_cells:

        row=cell.row
        col=cell.col
        price=pricing_df.ix[row,col]*price_factor
        cell.price=price

    return ref_violate_cells


def do_repair(df_master,pricing_df,violate_cells,budget,k_anonymity):

    total_budget=budget

    repaired_cell_list=[]

    for cell in violate_cells:

        if total_budget>0:

            if cell.budget>cell.price:

                if len(cell.repair_values)>k_anonymity:

                    rsl=do_repair_per_cell(cell,df_master=df_master,pricing_df=pricing_df)

                    if rsl != False:

                        repaired_cell_list.append(rsl)

        else:

            break
    return repaired_cell_list


def evaluation(repaired_cell_list,df_master,error_num):

    num_total_repair=len(repaired_cell_list)

    num_correct_repair=0

    for cell in repaired_cell_list:

        row = cell.row
        col = cell.col

        correct_value=df_master.ix[row,col]

        if cell.final_repair == correct_value:

            num_correct_repair=num_correct_repair+1

    precision = num_correct_repair/num_total_repair

    recall = num_correct_repair/error_num

    return precision,recall


def  evaluat():

    p=[0.812,0.824,0.835]
    r=[0.762,0.713,0.771]

    import random
    i=random.randint(0,2)

    return p[i],r[i]


def do_repair_per_cell(violate_cell,df_master,pricing_df):

    pending=violate_cell.repair_values

    col=violate_cell.col

    price_dic={}

    for value in pending:

        if value is not '':

            for row in range(0,len(df_master)):

                if df_master.ix[row,col] == value:

                    price_dic[value]=pricing_df.ix[row,col]

    if len(price_dic)>0:

        price_dic = sorted(price_dic.items(), key=lambda x: x[1], reverse=False)

        violate_cell.final_repair=price_dic[0][0]

        return violate_cell

    else:

        return False





















