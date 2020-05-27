import cvxpy as cv
from pandasql import sqldf

from src.data_cleaning import apply_update_cell
from src.data_cleaning import rollback
from src.ontology_node import *


def check_update_optimize(updateList, q, df):
    """
    Main idea: 为了优化, 不用把update 真的apply 到instance上, 而只用check update 是否更改了query answer 返回的那块数据
    Args:
        updateList, 包含了所有的updates
        q: 输入的一个query
        df: original database instance
    Return:
        a list of updates, 这些 update 生成的instance 和 original instance df 在query 下的 answer 不相等.
    """
    originalResult = get_query_result(q, df)
    unequalList = []
    for update in updateList:
        if check_change(update, originalResult, df):
            unequalList.append(update)
    return unequalList

## generate price without optimization (directly check whether the result of queries)
def check_update_direct(updateList, q, df):
    """
    Main idea: directly apply updates to instance, check whether the results are equal
    Args:
        updateList: a list of updates
        q: input query
        df: original database instance
    Return:
        a list of updates, whose query results are not equal to original instance 
        (这些 update 生成的instance 和 original instance df 在query 下的 answer 不相等)
    """

    df_old = df
    originalResult = get_query_result(q, df_old)
    unequalList = []
    for update in updateList:
        df_new = apply_update_cell(df_old, update)
        if not originalResult.equals(get_query_result(q, df_new)):
            unequalList.append(update)
        df_old = rollback(df_new, update)
    return unequalList


def check_change(update, originalResult, df):
    """
    Main idea:
        check whether the update is in the original result area
        这个是优化过的
    Args:
        update: one update
        originalResult: the answer of query on df, it should include the position (row, column) and the value information
        df: original database instance
    Return:
        True, if the update belongs to original result; otherwise, False
    """
    pass


def construct_constraints(updateList, df, queryList, pointList):
    """
    Main idea:
        construct the constraints of the convex optimization
    Args:
        updatelist: a list of update cell
        df: original df
        querylist: a list of queries
        pointList: a list of price point
    
    Return:
        a list of constraint, each of them is a tuple pair, which contain (unquealList, point)
        每个ids中, 是 a list of update id; 如果没有unequal的, 也就是说在这些update中, query的result都是equal的
        那么会输出一个emputy list
    """
    # query, point pairs
    pairwise = zip(queryList, pointList)

    # construct constraints


    constraintList = []
    for element in pairwise:
        unequalList = check_update_direct(updateList, element[0], df)
        ids = [i.get_id() for i in unequalList]
        constraintList.append((ids, element[1]))
    return constraintList


def calculate_weight(updateList, df, queryList, pointList):
    """
    Main idea:
        given a list of database instance (updateList), a bundle of queries and corresponding price points, calculate the weight
    for each instance in the support set (updateList)

    Args:
        updateList
        queryList, 用户指定
        pointList, 用户指定
        df, original database instance
    Return:
        a list of weights for each instance in the updateList
    """

    n = len(updateList)
    # W type is Variable, which is a inner type defined in cvxpy
    # W.value type is numpy matrix
    W = cv.Variable(n)

    cost = sum(cv.entr(W))
    obj = cv.Maximize(cost)

    # form the constraints
    constraints = []

    # the whole database price 不需要进入whole database price point， 因为有些地方是privacy的，无价的
    #     constraints.append(sum([W[i] for i in range(n)]) == databasePoint)
    constraints.append(0 <= W)

    constraintList = construct_constraints(updateList, df, queryList, pointList)
    for constraint in constraintList:
        index = constraint[0]
        value = constraint[1]
        if index is not None:
            constraints.append(sum([W[i] for i in index]) == value)

    # Form and solve problem.
    prob = cv.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", W.value)
    return W.value

def calculate_price(query, updateList, df_original, queryList, pointList):
    """
    Main idea: the price is computes as the weighted sum of the instances in the support set for which
    $Q(D) \ne Q(S)$. 
    :param query: the query that need for pricing
    :param updateList: to generate support set
    :param df_original: original instance
    :param queryList: the given query list for price point
    :param pointList: the price pointlist
    :return: a price for the given query
    """

    # calculate weights for each instance in the support set
    weights = calculate_weight(updateList, df_original, queryList, pointList)
    unequal_updates = check_update_direct(updateList, query, df_original)
    price = 0
    for u in unequal_updates:
        price = price + weights[u.get_id()]

    return price

def get_query_result(qry, df_local):
    '''
    这里不用传入 df, 是因为在qry中会指定df, 比如 from df 和 from df_new 是不一样的
    sqldf 会 lookup 全局 variable, 然后根据名字来确定是在哪个dataframe 中找
    Args:
        qry: a string with sqlite query format
    Return:
        result in the form of df
    '''

    df = df_local
    result = sqldf(qry, locals())

    return result


def get_general_query_result(query, df_local, root, level):
    """
    get the query result at the given level

    """

    # 首先得到ground level的 result， 这是一个 df 对象 （one column or one cell）
    groundResult = get_query_result(query, df_local)
    #     display(groundResult)

    # 把该result 先上 uplevel
    (row, column) = groundResult.shape
    for i in range(row):
        originalGroundValue = groundResult.iat[i, column - 1]
        #         print(originalGroundValue)
        groundNode = value_to_node(root, originalGroundValue)
        #         print (groundNode.getValue())
        geValue = get_node_by_level(root, groundNode, level).get_value()
        #         print (geValue)
        # replace ground value with general value
        groundResult.iat[i, column - 1] = geValue
    return groundResult
