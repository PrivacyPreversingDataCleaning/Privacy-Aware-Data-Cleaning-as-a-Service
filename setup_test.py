from experiment.accuracy_experiment import *

"""
这个用来测试, 在使用master 前, 是否进行了初步的clean
如果发现了violation, 就把value 换成当前pattern下最先发现的那个row的value
但理想情况是, 应该换成当前pattern下, frequency 最高的那个value
"""
master_path = './master_test_data.csv'
fd_path = './fd_test.txt'


def check_master_test():
    df_master, fd_list = load_master(master_path, fd_path)
    print(df_master)
    print(df_master[df_master.columns[3]].value_counts().index[0])
    df_master = check_df(df_master, fd_list)
    print(df_master)
    for f in fd_list:
        print(f.lhs, f.rhs)



def generate_target_test():
    """
    generate target test
    这里有个小问题, 随机的value 应该是不包含原始value的
    """
    df_master, fd_list = load_master(master_path, fd_path)
    df_master = check_df(df_master, fd_list)
    df_target, update_cell_list = generate_target(df_master, fd_list, error_num=2, percentage=0.5)
    print(df_target)
    for c in update_cell_list:
        print(str(c))


generate_target_test()

def calculate_error_factor_test():
    """
    test find_violation, calculate_error_factor
    :return:
    """
    df_master, fd_list = load_master(master_path, fd_path)
    df_master = check_df(df_master, fd_list)
    df_target, update_cell_list = generate_target(df_master, fd_list, error_num=2, percentage=0.5)
    find_violate_cells(fd_list=fd_list, df_target=df_target)