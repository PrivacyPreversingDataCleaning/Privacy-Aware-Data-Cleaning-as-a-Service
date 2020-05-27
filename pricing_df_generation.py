# 如何分配price给每个cell:
# 给每个column 一个weight, sensitive的column weight 高, 一般的 column weight 低.
# 这样就给每个column 分配了一个price.
# 然后, 根据每个column中的value的 statistics, 再来分配price.
# 比如column A中, 有 不同的 distinct value 10 个, 但每个value出现了很多次. 那么出现次数少的, price 应该高,
# 出现次数多的, price 应该低.  这样之后, 给定一个df_master 和一个总的price,
# 我们就得到了一个price table, 表明了每个cell的price.
#
# 写一个方法, calculate_master_cell_price, 输入是 df_master, sensitive_column, total_price, 输出是一个df表,
# 但内容不是原始的master中的cell value, 而是 该cell 的price.
import copy


def calculate_price(sensitive_weight,distinct_weight):
    '''

    :param sensitive_weight:
    :param distinct_weight:
    :return:
    '''

    price=distinct_weight*sensitive_weight

    return price

def calculate_column_weight(col_weight_dict):
    """
    计算每个column的weight.
    :param col_weight_dict: 
    :return: 
    """
    ref_col_weight_dict = col_weight_dict
    total = 0
    for item in ref_col_weight_dict:
        total = total + ref_col_weight_dict[item]
    for item in ref_col_weight_dict:
        ref_col_weight_dict[item] = ref_col_weight_dict[item]/total
    return ref_col_weight_dict


def generate_pricing_df_per_col(df_master,column,column_weight):
    '''

    :param df_master:
    :param column_name:
    :param column_weight:
    :return:
    '''

    col=column

    col_df=copy.deepcopy(df_master[df_master.columns[col]])

    distinct_values_dict=create_distinct_values_dict(df_master,column)

    distinct_size=len(df_master)

    cell_price_tmp_list=[]

    cell_price_saving_list = []

    total_price = 0

    for row in range(0,len(col_df)):

        value=col_df.ix[row,0]

        frequency=distinct_values_dict[value]

        distinct_weight=distinct_size/frequency

        price=calculate_price(column_weight,distinct_weight)

        total_price=total_price+price

        cell_price_tmp_list.append(price)

    for price in cell_price_tmp_list:

        final_price=(price/total_price)*column_weight

        cell_price_saving_list.append(final_price)

    return cell_price_saving_list


def generate_pricing_df(df_master,columns_weight_dic):
    '''

    :param df_master:
    :param columns_weight_dic:
    :return:
    '''
    pricing_df=copy.deepcopy(df_master)

    for col in range(0,len(pricing_df.columns)):

        pricing_df_col=generate_pricing_df_per_col(df_master=pricing_df,column=col,column_weight=columns_weight_dic[col])

        for row in range(0,len(pricing_df)):

            pricing_df.ix[row,col]=pricing_df_col[row]

    return pricing_df


def create_distinct_values_dict(df_master,column_index):
    '''

    :param df_master:
    :param column_name:
    :return: a dict, key: distinct_value, value: it's frequency
    '''

    col=column_index

    distinct_value_dict={}

    for row in range(0,len(df_master)):

        value=df_master.ix[row,col]

        if value in distinct_value_dict:

            distinct_value_dict[value]=distinct_value_dict[value]+1

        else:

            distinct_value_dict[value]=1

    return distinct_value_dict





































