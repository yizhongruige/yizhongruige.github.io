import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

#file = open(r"D:\python\yichuansuanfa\68门\dolpin_1_1.txt")
#file=open(r"D:\python\yichuansuanfa\football\football.txt")
file=open(r"D:\python\yichuansuanfa\football\pobooks.txt")
prime = []
prime2=[]
data = file.readlines()
for line in data:
    line = line.strip('\n')
    prime.append(line)
    prime2.append(list(line))
G = nx.Graph()
Matrix=np.array(prime2)
print(len(Matrix))
print((Matrix.shape))

ls={}
for i in range(len(Matrix)):
    G.add_node(i, desc=i)
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if int(Matrix[i][j]) == 1:
            G.add_edge(i, j)
plt.show()
for i in range(len(Matrix)):
    ls[i]=G.degree(i)
print(ls)
LS=list(ls.items())
LS.sort(key=lambda x: x[1], reverse=True)
    #print('度最大的点是{}，有{}个邻点\n度次点是{}，有{}个邻点'.format(LS[0][0], LS[0][1], LS[1][0], LS[1][1]))
for i in range(8):

    print('度最大的点是{}，有{}个邻点'.format(LS[i][0],LS[i][1]))
