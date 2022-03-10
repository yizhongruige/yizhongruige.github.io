fo=open(r"D:\python\yichuansuanfa\68门\Adj_1.txt",'r')
fi =open(r"D:\python\yichuansuanfa\68门\Adj_1_1.txt",'w')
b=fo.readlines()
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

file = open("D:\python\yichuansuanfa\Adj.txt")
prime = []
prime2=[]
data = file.readlines()
for line in data:
    line = line.strip('\n')
    prime.append(line)
    prime2.append(list(line))
G = nx.Graph()
Matrix=np.array(prime2)
print(Matrix)
print(type(Matrix[1][2]))
print((Matrix.shape))

for i in range(len(Matrix)):
    G.add_node(i, desc=i)
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if int(Matrix[i][j]) == 1:
            G.add_edge(i, j)

ls=[]
ls1=[]
pos = nx.spring_layout(G)
nx.draw(G, pos)
node_labels = nx.get_node_attributes(G, 'desc')
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.show()
for i in range(68):
    if nx.has_path(G, i, 30):
        ls.append(i)
print('nihao ')
print(ls)#ls是我需要的行列索引号
for i in ls:
    ls1.append(prime2[i])#将对应行取出来成为ls1
Matrix2=np.array((ls1))#将一维列表变成数组
c=Matrix2[:,ls]#将数组中我需要的列取出来
#list(c)
ls2=[]
#引文write只能写入字符串所以将数组变成字符串
for i in range(66):
    str1 = ''
    for j in range(66):
        str1+=c[i][j]
    ls2.append(str1+'\n')
#ls2是由多个字符串形成的列表
print(c)
print((c.shape))
print(ls2)
for i in ls2:
    fi.write(i)
fo.close()
fi.close()
