import random
from copy import deepcopy

from src.data_error_rank import read_fd
from src.data_io import get_fd_columns


class UpdateCell():
    def __init__(self,row,column,value):
        self.row=row
        self.column=column
        self.old_value=value
        self.update_value=None

def generate_target(df_master, fd_list,error_num, percentage):

    '''
    随机生成一些cell的坐标, 把他们的value 换成该列其他的随机值,同时要保留以前的 value 信息,
    作为 ground truth. percentage 控制master的前 百分之多少row作为 target. 可以用到之前的update_cell中的方法
    :param df_master:
    :param error_num:
    :param percentage:
    :return:
    '''

    master=deepcopy(df_master)
    fdcolumns=get_fd_columns(fd_list)
    df_size=len(master)
    target_size=int(df_size*percentage)
    update_df=master.iloc[:target_size]
    updatecells_list=[]

    for num in range(0,error_num):
        random_col= random.choice(fdcolumns)
        random_row=random.randint(0,target_size-2)

        value=update_df.iat[random_row,random_col]
        uc=UpdateCell(random_row,random_col,value)
        updaterow=random.randint(0,len(master)-1)
        update_df.iat[random_row,random_col]=master.iat[updaterow,random_col]
        uc.update_value = master.iat[updaterow, random_col]
        updatecells_list.append(uc)

    return update_df,updatecells_list

