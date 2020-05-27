from src.data_matching import find_matches


def query_generator(violate_cell, matched_df, geColumn):
    """
    generate a list of query strings for pricing.
    :param violate_cell: a list of general values
    :param matched_df:
    :param rows_index:
    :param geColumn: dictionary, key is column index, and value is the root node object.
    :return: a list of tuples (str_query, matched_df)
    """
    query_df_list = []

    for GV in violate_cell:

        rows_index = find_matches(GV, matched_df, geColumn, threshold=0.8)
        col = GV.col
        # 根据返回的matched row_index, 切割matched_row, 组成新的 matched_df
        str_query = 'select ' + str(matched_df.columns[col]) + ' from ' + 'df'
        query_df_list.append((str_query, matched_df))

    return query_df_list


