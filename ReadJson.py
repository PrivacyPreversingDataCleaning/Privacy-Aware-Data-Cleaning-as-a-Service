from src.OntologyNode import OntologyNode
from src.OntologyNode import getFamily
def loadJson(filepath):
   '''
   读取json文件
   :param filepath: 测
   :return: 数据字典 dic
   '''
   import json
   fp=filepath
   f= open(fp)
   dic = json.load(f)
   return dic

def deJson2Tree(dic,root):
    '''
    递归解析json结构
    :param dic:
    :param root:
    :return: 以root为根的树
    '''

    if "value" in dic:

        p =dic['value']
        rp=OntologyNode(p)
        root.addChild(rp)
        rp.parent=root
        children = dic['children']
        for child in children:
            if "value" in child:
                deJson2Tree(child,rp)

def reJson(filepath):
    '''
    整合上述方法，返回以root为根的树
    example：
    country=reJson('../data/lTest/country.json')
    :param filepath:
    :return:
    '''
    root=OntologyNode('root')
    deJson2Tree(loadJson(filepath),root)
    return root


# from src.OntologyNode import findNodes
# a= reJson('../data/lTest/city.json')
#
# b=findNodes(a,'NB')
#
# print(b.getParent())
