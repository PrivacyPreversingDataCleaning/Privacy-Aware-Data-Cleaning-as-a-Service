import configparser
import os

from src.ontology_node import OntologyNode


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
    ontology_dict=scan_ontology_folder(folder_path)
    for tree in ontology_dict:
        file_path=ontology_dict[tree]
        ontology_dict[tree]=read_json(file_path)
    return ontology_dict


def tidy_df(partitiondf,partition,columns):
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


def scan_ontology_folder(folder_path):
    pathDir =  os.listdir(folder_path)
    ontology_dict={}

    for file_name in pathDir:
        file_path=folder_path+'/'+file_name
        ontology_dict[file_name.split('.')[0]]=file_path

    return ontology_dict


def get_fd_columns(fd_list):
    columns=[]

    for fd in fd_list:

        columns=columns+fd.rhs+fd.lhs

    columns=list(set(columns))

    return columns



