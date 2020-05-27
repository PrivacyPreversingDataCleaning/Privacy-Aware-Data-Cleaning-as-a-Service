from src.OntologyNode import OntologyNode


class OntologyTree():
    def __init__(self, root):
        self.root = root
        self.children = 'null'

    def setRoot(self, ro):
        self.root = ro

    def getRoot(self):
        return self.root

    def setChildren(self, c):
        self.children = c

    def getChildren(self):
        return self.children

    def addChild(self,child,parent):

        child.setParent(parent)


# def findNodes(root,p):  # value to object
#
#     ro=root
#     if ro.value==p :
#        return ro
#
#     else:
#         list=ro.children
#         for l in list:
#             tmp = findNodes(l, p)
#             if tmp is not None:
#                 return tmp
#
# def getParents(root,p,li):     #object to value
#     list=li
#     if p.parent is not 'null':
#         a=p.parent
#         list.append(a.value)
#         getParents(root,a,list)
#         return list
#     else:
#         return list
#
# def getChildren(root,p,li):    #object to value   including itself
#     list=li
#     if p.isLeaf() is not 'true':
#         if p.value not in list:
#            list.append(p.value)
#         a=p.children
#         for c in a:
#             list.append(c.value)
#             getChildren(root,c,list)
#         return list
#     else:
#         if p.value not in list:
#            list.append(p.value)
#
#         return list
#
# def getFamily(root,p):
#     listC=[]
#     listC=getChildren(root,p,listC)
#     listP=[]
#     listP=getParents(root,p,listP)
#     list=listC+listP
#     return list
