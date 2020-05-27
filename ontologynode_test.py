from src.ontology_node import *

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

b.set_parent(a)
a.add_child(b)
c.set_parent(a)
a.add_child(c)
d.set_parent(b)
b.add_child(d)
e.set_parent(b)
b.add_child(e)
k.set_parent(b)
b.add_child(k)
f.set_parent(d)
d.add_child(f)
g.set_parent(d)
d.add_child(g)
h.set_parent(e)
e.add_child(h)
i.set_parent(e)
e.add_child(i)
j.set_parent(e)
e.add_child(j)
l.set_parent(k)
k.add_child(l)
m.set_parent(k)
k.add_child(m)
n.set_parent(k)
k.add_child(n)

root = a

print(' '.join(get_ancestors(root, f)))
print(' '.join(get_all_leafnodes(root, f)))
print(' '.join(get_all_descendants(root, d)))
# print(getUpper(root,k,list))
l = get_all_descendants(root, d)
print('l is :', ' '.join([e for e in l]))
assert b.get_level() == 1  # b is Canada
assert value_to_node(a, "ON") == d  # d is ON

assert get_node_by_level(root, f, 2).get_value() == "ON"  # f is Hamilton


print('')
print(str(a))
