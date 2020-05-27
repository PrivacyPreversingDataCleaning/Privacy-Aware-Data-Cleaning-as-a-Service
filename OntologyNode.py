class OntologyNode():
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = 'null'

    def getValue(self):
        return self.value

    def setValue(self, v):
        self.value = v

    def isLeaf(self):
        tag = 'false'
        if len(self.children) == 0:
            tag = 'true'
        return tag

    def setParent(self, parent):
        self.parent = parent
        self.parent.children.append(self)

    def getParent(self):
        return self.parent

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self):
        return self.children

    def getLeafNodes(self):

        result = []

        if self.isLeaf() == 'true':
            result.append(self.toString())

        else:
            children = self.getChildren()
            for child in children:
                result.append(child.getLeafNodes())
        return result  ###return values

    def getLeafNodeValues(self):  ####invalid
        leafNodes = self.getLeafNodes()
        result = []
        for leafNode in leafNodes:
            result.append(leafNode[0].value)
        return result

    def toString(self):
        return self.value

    def getHeight(self):
        if self.parent == 'null':
            return 0
        else:
            return self.parent.getHeight()+1

    def getRoot(self):
        if self.parent == 'null':
            return self.toString()

        else:
            up = self.getParent()
            return up.getRoot()

def findNodes(root,p):  # value to object

    ro=root
    if ro.value==p :
       return ro

    else:
        list=ro.children
        for l in list:
            tmp = findNodes(l, p)
            if tmp is not None:
                return tmp

def getParents(root,p,li):     #object to value
    list=li
    if p.parent is not 'null':
        a=p.parent
        list.append(a.value)
        getParents(root,a,list)
        return list
    else:
        return list

def getChildren(root,p,li):    #object to value   including itself
    list=li
    if p.isLeaf() is not 'true':
        if p.value not in list:
           list.append(p.value)
        a=p.children
        for c in a:
            list.append(c.value)
            getChildren(root,c,list)
        return list
    else:
        if p.value not in list:
           list.append(p.value)

        return list

def getFamily(root,p):
    listC=[]
    listC=getChildren(root,p,listC)
    listP=[]
    listP=getParents(root,p,listP)
    list=listC+listP
    return list

# ------------

def getDescendantNodes(root,p,li):
    list = li
    if p.isLeaf() is not 'true':
        a = p.children
        for c in a:
            getChildren(root, c, list)
        return list
    else:
        list.append(p.value)

def getDomainsize(root):
    list=getFamily(root,root)
    size=len(list)
    return size

def getLeaves(root,p):

    list = []
    rsl=[]
    if p.isLeaf() is not 'true':
        a = p.children
        for c in a:
            getChildren(root, c, list)

        for var in list:

            node=findNodes(root,var)
            if node.isLeaf()=='true':
                rsl.append(var)
        return rsl
    else:
        list.append(p.value)
        return list

def getUpper(root,k,li):   #with itself

    list=getParents(root,k,li)

    list.append(k.value)

    return list

a = OntologyNode('North America')
b = OntologyNode('Canada')
c = OntologyNode('America')
d = OntologyNode('ON')
e = OntologyNode('BC')
f = OntologyNode('Hamilton')
g = OntologyNode('Toronto')
h = OntologyNode('Vancouver')
i = OntologyNode('Kamloops')
j = OntologyNode('Penticton')
k = OntologyNode('NB')
l = OntologyNode('Baker Brook')
m = OntologyNode('Bathurst')
n = OntologyNode('Dieppe')


b.setParent(a)
c.setParent(a)
d.setParent(b)
e.setParent(b)
k.setParent(b)
f.setParent(d)
g.setParent(d)
h.setParent(e)
i.setParent(e)
j.setParent(e)
l.setParent(k)
m.setParent(k)
n.setParent(k)


root=a


# print(getChildren(root,b,list))
# print(getDescendantNodes(root,n,list))

list=[]
# print(getParents(root,k,list))

def getUpper(root,k,li):   #with itself

    list=getParents(root,k,li)

    list.append(k.value)

    return list


# print(getUpper(root,k,list))

