from src.ontology_node import get_all_leafnodes
from src.ontology_node import value_to_node


def find_matching_rows(violate_cell,record, df_master,threshold):

    """
    given a record from target, find out all the matched record records in master.
    return the row index in the df_master
    :param record:
    :param df_master:
    :return: a list of matched records
    """
    row_index = []
    columns = df_master.columns
    columns_size=len(columns)-1
    row_dic={}
    minimal_true=int(columns_size*threshold)

    for i in range(0, len(df_master)):
        row_dic[i]=0

    for i in range(0, len(columns)):
        if i != violate_cell.col:
            column_check = df_master[columns[i]].isin(record[i])
            for k in range(0, len(column_check)):
                if column_check[k] == True:
                    row_dic[k]=row_dic[k]+1

    for row in row_dic:
        if row_dic[row]>=minimal_true:
            row_index.append(row)
    return row_index


def matched_df_generator(df_master, rows_index):
    matched_df = df_master.iloc[rows_index]
    return matched_df


def find_matches(violate_GeValue, df_master, geColumn,threshold):

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

    return find_matching_rows(violate_GeValue,record, df_master,threshold)


def find_matched_df_ground(violate_cell,df_target,df_master,threshold):
    """
    把需要match的cell所在的row，除了cell所在列，其他所有的列的数据已list的形式存入一个list。如果是general value，存本身以及它之下所有的ground value。
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

    row_index=find_matching_rows(violate_cell,record, df_master,threshold)
    matched_df=matched_df_generator(df_master, row_index)

    return matched_df

