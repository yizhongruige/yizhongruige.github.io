import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from networkx.algorithms import community
pop_size = 100
DNA_SIZE =66
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.003
N_GENERATIONS = 100
number_division=2
n=0
ls1=[]
ls2=[]
ls3=[]
pop = []
generation = 0
label=0
low_threshold=0.4*DNA_SIZE
comunity_jiezhi=[]

file = open(r"D:\python\yichuansuanfa\68门\Adj_1_1.txt",'r')
prime = []
high_betweenness=[47]
data = file.readlines()
for line in data:
    line = line.strip('\n')
    prime.append(line)
prime1 = [[0 for i in range(DNA_SIZE)] for j in range(DNA_SIZE)]
for i in range(0, DNA_SIZE):
    for j in range(0, DNA_SIZE):
        if int(prime[i][j]) == 1:
            prime1[i][j] = 1
        else:
            prime1[i][j] = 0
for i in range(0, DNA_SIZE):
    for j in range(0, DNA_SIZE):
        if prime1[i][j] == 1:
            prime1[j][i] = 1
print(prime1)
G = nx.Graph()

Matrix = np.array(prime)
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if int(Matrix[i][j]) == 1:
            G.add_edge(i, j)
            G.add_node(i, desc=i)
            G.add_node(j, desc=j)
p=pos=nx.spring_layout(G)
nx.draw(G, pos)  # 这里注意，这里默认了nodecolor为BLUE了。
node_labels = nx.get_node_attributes(G, 'desc')
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.show()


for i in range(len(Matrix)):
    G.add_node(i, desc=i)

for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if int(Matrix[i][j]) == 1:
            G.add_edge(i, j)

LS={}
'''for i in range(DNA_SIZE):
    LS[i]=[]
    for j in range(DNA_SIZE):
        if prime1[i][j]==1:
            LS[i].append(j)
print(LS)


def BFS(Central_node, LS):
    queue = []
    visit = []
    flag=0
    count=0
    queue.append(Central_node)
    visit.append(Central_node)
    while queue:
        node = queue.pop(0)
        nodes = LS[node]
        for i in nodes:
            if i in comunity_jiezhi:
                count=len(visit)
                if count>low_threshold:
                    flag=1
                    break
            elif i not in visit:
                queue.append(i)
                visit.append(i)

        if flag==1:
            return visit'''
def BFS(Central_node, G):
    ls = G.nodes
    LS = {}
    for i in ls:
        LS[i] = G.neighbors(i)
    print(ls)
    print(LS)
    queue = []
    visit = []
    flag=0
    count=0
    queue.append(Central_node)
    visit.append(Central_node)
    while queue:
        node = queue.pop(0)
        nodes = LS[node]
        for i in nodes:
            if i in high_betweenness:
                count=len(visit)
                if count>low_threshold:
                    flag=1
                    break
            elif i not in visit:
                queue.append(i)
                visit.append(i)

        if flag==1:
            return visit

def get_fitness(pop):
    f = []
    fitness=[]
    fitness1=[]
    for gene in pop:
        a = []
        b = []
        c = []
        d = []
        for i in range(DNA_SIZE):
            if gene[i] ==0:
                a.append(i)
            if gene[i]==1:
                b.append(i)

        d.append(a)
        d.append(b)
        d.append(c)
        num_cut = nx.cut_size(G, a, b)
        #print(num_cut)
        #shuchushow1(d)

        f.append(num_cut)
    print(f)
    fitness=f
    sum1 = sum(fitness)
    a1 = []
    for i in range(0, len(fitness)):
        a1.append(fitness[i] / sum1)
    print(a1)
    f_max = max(f)
    f_min = min(f)
    x,y=f.index(f_min),f.index(f_max)
    print(x,y)
    for i in fitness:
        fitness1.append(f_max-i)
    b1 = []
    sum1=sum(fitness1)
    if sum1==0:
        for i in range(len(fitness)):
            b1.append(1)
    else:

        for i in range(0, len(fitness1)):

            b1.append(fitness1[i] / sum1)
        print(b1)

        print('最坏的cut值为',f[y],'在方法调整前后的选中概率变化为：',a1[y],b1[y])
        print('最好的的cut值为', f[x], '在方法调整前后的选中概率变化为：', a1[y], b1[x])
    print(f_max, f_min)
    if f_max == f_min:
        global label
        label=1
        return label
    else:
        return fitness1
    '''else:
        for i in range(0, len(f)):
            fitness.append((f_max - f[i]) + (f_max - f_min) / 3)
        print(fitness)'''


def select(pop, fitness):
    sum1 = sum(fitness)
    a = []
    new_pop = []
    b = list(range(0, pop_size))
    for i in range(0, len(fitness)):
        a.append(fitness[i] / sum1)
    print(sum(a))
    pop_choice = np.random.choice(a=b, size=pop_size, replace=True, p=a)
    print(pop_choice)
    for i in pop_choice:
        new_pop.append(pop[i])
    return new_pop

def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    if len(pop[0])>L_DNA:
        pop = undress_set(pop)
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(pop_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=L_DNA)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop

def bianyi(i):
    if i==0:
        return 1
    if i==1:
        return 0


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, L_DNA)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = int(bianyi((child[mutate_point]))) # 将变异点的二进制

def shuchushow1(d):
    G = nx.Graph()
    Matrix = np.array(prime)
    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if int(Matrix[i][j]) == 1:
                G.add_edge(i, j)
                G.add_node(i, desc=i)
                G.add_node(j, desc=j)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)  # 这里注意，这里默认了nodecolor为BLUE了。
    nx.draw_networkx_nodes(G, pos, nodelist=d[0], node_color='red')
    nx.draw_networkx_nodes(G, pos, nodelist=d[1], node_color='green')
    nx.draw_networkx_nodes(G, pos, nodelist=d[2], node_color='grey')
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.show()

def shuchushow(pop):
    a = []
    b=[]
    c=[]
    new_components=[]
    for i in range(0, len(pop[0])):
        if pop[0][i] == 0:
            a.append(i)
    for i in range(0, len(pop[0])):
        if pop[0][i] == 1:
            b.append(i)

    print('标红的节点为：',a)
    print('有',len(a),'个')
    print('标绿的节点为：',b)
    print('有',len(b),'个')
    print('标灰色的节点为：',c)
    print('有',len(c),'个')

    G = nx.Graph()
    Matrix = np.array(prime)
    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if int(Matrix[i][j]) == 1:
                G.add_edge(i, j)
                G.add_node(i, desc=i)
                G.add_node(j, desc=j)
    pos =p #nx.spring_layout(G)
    nx.draw(G, pos)  # 这里注意，这里默认了nodecolor为BLUE了。
    nx.draw_networkx_nodes(G, pos, nodelist=a, node_color='red')
    nx.draw_networkx_nodes(G, pos, nodelist=b, node_color='green')
    nx.draw_networkx_nodes(G, pos, nodelist=c, node_color='grey')
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.show()

def dress_set(pop,ls2,ls1):
    pop_new=[]
    for gene in pop:
        number_division = 2
        gene1=[]
        n=0
        m=0
        for i in range(DNA_SIZE):
            gene1.append(0)
        for i in ls2:
            gene1[i]=gene[n]
            n += 1
        while number_division>0:
            for i in ls1[m]:
                gene1[i]=m
            m += 1
            number_division-=1
        pop_new.append(gene1)
    pop=pop_new
    return pop

def undress_set(pop):
    pop_new=[]
    for gene in pop:
        gene1=[]
        for i in ls2:
            gene1.append(gene[i])
        pop_new.append(gene1)
    pop = pop_new
    return pop

def max_degree_search(G):
    ls={}
    node_labels = nx.get_node_attributes(G, 'desc')
    print(node_labels)

    for i in range(len(node_labels)):
        ls3=node_labels.values()
    print(ls3)
    for i in range(len(Matrix)):
        if i in ls3:
            ls[i] = G.degree(i)
        else:
            ls[i]=0
    print(ls)
    LS = list(ls.items())
    LS.sort(key=lambda x: x[1], reverse=True)
    return LS[0][0]

'''a=nx.betweenness_centrality(G)
ls=list(a.items())
ls.sort(key=lambda x:x[1],reverse=True)
for i in range(2):
    print('边介数最大的是{}为{}'.format(ls[i][0],ls[i][1]))
    comunity_jiezhi.append(ls[i][0])'''

while number_division>0:
    list1=[]
    n+=1
    Central_node=max_degree_search(G)
    print(Central_node)
    list1=BFS(Central_node,G)
    print('第',n,'次聚类点集为：',list1)
    ls1.append(list1)
    G.remove_nodes_from(list1)
    number_division-=1
print('两次次聚类的点集合为:',ls1)
node_labels = nx.get_node_attributes(G, 'desc')
for i in range(len(node_labels)):
    ls2=node_labels.values()





G = nx.Graph()
Matrix = np.array(prime)
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if int(Matrix[i][j]) == 1:
            G.add_edge(i, j)
            G.add_node(i, desc=i)
            G.add_node(j, desc=j)
#G.add_edge(14, 30)
pos = p#nx.spring_layout(G)
print(pos)
nx.draw(G, pos)  # 这里注意，这里默认了nodecolor为BLUE了。
nx.draw_networkx_nodes(G, pos, nodelist=ls1[0], node_color='red')
nx.draw_networkx_nodes(G, pos, nodelist=ls1[1], node_color='green')

node_labels = nx.get_node_attributes(G, 'desc')
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.show()

'''l_ls2=len(ls2)
print(l_ls2)
print(ls2)
ls_yi=[]
ls2_xin=[]
for i in ls2:
    num=0
    print(i)
    ll = len(list(G.neighbors(i)))
    print(i)
    for each in list(G.neighbors(i)):

        if each in ls1[0]:
            num+=1
            #print(num)
        if each in ls1[1]:
            num-=1
            #print(num)
    print(list(G.neighbors(i)), ll,i)
    if num==ll:
        ls1[0].append(i)
        ls_yi.append(i)
    if num==(-ll):
        ls1[1].append(i)
        ls_yi.append((i))
for i in ls2:
    if i not in ls_yi:
        ls2_xin.append(i)
ls2=ls2_xin
L_DNA=len(ls2)
print(ls2,ls1)
pos = p#nx.spring_layout(G)
print(pos)
nx.draw(G, pos)  # 这里注意，这里默认了nodecolor为BLUE了。
nx.draw_networkx_nodes(G, pos, nodelist=ls1[0], node_color='red')
nx.draw_networkx_nodes(G, pos, nodelist=ls1[1], node_color='green')

node_labels = nx.get_node_attributes(G, 'desc')
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.show()'''
number_division=2

while pop_size>0:
    gene_old= np.ones(L_DNA)
    gene_old[:int(L_DNA/number_division)] = 0
    np.random.shuffle(gene_old)
    pop_size -= 1
    pop.append(gene_old)
print(pop)

pop_size=100

for _ in range(N_GENERATIONS):  # 种群迭代进化N_GENERATIONS代
    pop = crossover_and_mutation(pop, CROSSOVER_RATE)
    pop=dress_set(pop,ls2,ls1)# 种群通过交叉变异产生后代
    fitness = get_fitness(pop) # 对种群中的每个个体进行评估
    if label==1:
        break
    pop = select(pop, fitness)# 选择生成新的种群
    generation += 1
    print('第', generation, '代')
    print(pop[0], pop[1])

shuchushow(pop)
#这个版本相对于第一版改动了，遗传算法中对于fitness的表达，因为之前收敛效果不好，我将种群中所有的gene的fitness都减去最小的那个作为该gene的fitness.
