from src.OntologyNode import findNodes
from src.OntologyNode import getDescendantNodes
from src.OntologyNode import getDomainsize
import math

def cp_g(value,root):
    '''
    Certainty Penalty for Ontology attribute
    :param value: the value of one tuple's one attribute
    :param root: attribute's tree's root
    :return: result
    '''

    node=findNodes(root,value)
    if(node.isLeaf() == 'true'):
        result=0
    else:
        list=[]
        list=getDescendantNodes(root,node,list)
        size=len(list)
        domain=getDomainsize(root)
        result=size/domain

        return result



def cp(tuple,list_tree,list_weight):
    '''
    for one tuple
    :param tuple:     list for values
    :param list_tree: list for attributes' nodes
    :param list_weight:list for attributes' weights
    :return:
    '''
    result=0
    for i in range(0,len(tuple)):

        r=cp_g(tuple[i],list_tree[i])*list_weight[i]

        result=result+r

    return result

def sp_g(value,root):
    '''
    Specificity Penalty for Ontology attribute
    :param value: the value of one tuple's one attribute
    :param root: attribute's tree's root
    :return: result
    '''

    node=findNodes(root,value)
    if(node.isLeaf() == 'true'):
        result=0
    else:
        list=[]
        list=getDescendantNodes(root,node,list)
        size=len(list)
        result=math.log(size)
        return result
