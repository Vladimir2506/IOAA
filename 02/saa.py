import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_city', type=int, action='store', default=5, help='Number of cities.')
parser.add_argument('--twenty', action='store_true', default=False, help='Show stats of 20 attempts.')
args = parser.parse_args()
 
def coordinate_init(size):
    #产生坐标字典
    coordinate_dict = {}
    coordinate_dict[0] = (0, 0)#起点是（0，0）
    for i in range(1, size + 1):#顺序标号随机坐标
        coordinate_dict[i] = (np.random.uniform(0, 1) * 100, np.random.uniform(0, 1) * 100)
    coordinate_dict[size + 1] = (0, 0)#终点是（0,0)
    #pdb.set_trace()
    return coordinate_dict
 
def distance_matrix(coordinate_dict,size):#生成距离矩阵
    d=np.zeros((size+2,size+2))
    for i in range(size+1):
        for j in range(size+1):
            if(i==j):
                continue
            if(d[i][j]!=0):
                continue
            x1 = coordinate_dict[i][0]
            y1 = coordinate_dict[i][1]
            x2 = coordinate_dict[j][0]
            y2 = coordinate_dict[j][1]
            distance=np.sqrt((x1-x2)**2+(y1-y2)**2)
            if(i==0):
                d[i][j]=d[size+1][j]=d[j][i]=d[j][size+1]=distance
            else:
                d[i][j]=d[j][i]=distance
    return d
 
def path_length(d_matrix,path_list,size):#计算路径长度
    length=0
    for i in range(size+1):
        length+=d_matrix[path_list[i]][path_list[i+1]]
    return length
 
def new_path(path_list,size):
    #二交换法
    change_head = np.random.randint(1,size+1)
    change_tail = np.random.randint(1,size+1)
    if(change_head>change_tail):
        change_head,change_tail=change_tail,change_head
    change_list = path_list[change_head:change_tail + 1]
    change_list.reverse()#change_head与change_tail之间的路径反序
    new_path_list = path_list[:change_head] + change_list + path_list[change_tail + 1:]
    return change_head,change_tail,new_path_list
 
def diff_old_new(d_matrix,path_list,new_path_list,head,tail):#计算新旧路径的长度之差
    old_length=d_matrix[path_list[head-1]][path_list[head]]+d_matrix[path_list[tail]][path_list[tail+1]]
    new_length=d_matrix[new_path_list[head-1]][new_path_list[head]]+d_matrix[new_path_list[tail]][new_path_list[tail+1]]
    delta_p=new_length-old_length
    return delta_p
 
 
T_start=1e6#起始温度
T_end=1e-6#结束温度
a=0.995#降温速率
Lk=50#内循环次数,马尔科夫链长
city=args.num_city
size=city-1
coordinate_dict=coordinate_init(size)
d=distance_matrix(coordinate_dict,size)#距离矩阵的生成
if not args.twenty:
    T = T_start
    path_list=list(range(size+2))#初始化路径
    best_path=path_length(d,path_list,size)#初始化最好路径长度
    print('Init Length = ',best_path)
    best_path_temp=[]#记录每个温度下最好路径长度
    best_path_list=[]#用于记录历史上最好路径
    balanced_path_list=path_list#记录每个温度下的平衡路径
    balenced_path_temp=[]#记录每个温度下平衡路径(局部最优)的长度
    while T>T_end:
        for i in range(Lk):
            head, tail, new_path_list = new_path(path_list, size)
            delta_p = diff_old_new(d, path_list, new_path_list, head, tail)
            if delta_p < 0:#接受状态
                balanced_path_list=path_list = new_path_list
                new_len=path_length(d,path_list,size)
                if(new_len<best_path):
                    best_path=new_len
                    best_path_list=path_list
            elif np.random.random() < np.exp(-delta_p / T):#以概率接受状态
                path_list = new_path_list
        path_list=balanced_path_list#继承该温度下的平衡状态（局部最优）
        T*=a#退火
        best_path_temp.append(best_path)
        balenced_path_temp.append(path_length(d,balanced_path_list,size))
    print('Best Length = ',best_path)
    x=[]
    y=[]
    for point in best_path_list:
        x.append(coordinate_dict[point][0])
        y.append(coordinate_dict[point][1])
    sns.set()
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(1)
    plt.plot(best_path_temp)#每个温度下最好路径长度
    plt.title('目标函数下降曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('距离')
    plt.figure(2)
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.title('最优路径图')
    plt.show()
else:
    best_lens = np.zeros([20])
    for t in range(20):
        T = T_start
        path_list=list(range(size+2))#初始化路径
        best_path=path_length(d,path_list,size)#初始化最好路径长度
        best_path_temp=[]#记录每个温度下最好路径长度
        best_path_list=[]#用于记录历史上最好路径
        balanced_path_list=path_list#记录每个温度下的平衡路径
        balenced_path_temp=[]#记录每个温度下平衡路径(局部最优)的长度
        while T>T_end:
            for i in range(Lk):
                head, tail, new_path_list = new_path(path_list, size)
                delta_p = diff_old_new(d, path_list, new_path_list, head, tail)
                if delta_p < 0:#接受状态
                    balanced_path_list=path_list = new_path_list
                    new_len=path_length(d,path_list,size)
                    if(new_len<best_path):
                        best_path=new_len
                        best_path_list=path_list
                elif np.random.random() < np.exp(-delta_p / T):#以概率接受状态
                    path_list = new_path_list
            path_list=balanced_path_list#继承该温度下的平衡状态（局部最优）
            T*=a#退火
            best_path_temp.append(best_path)
            balenced_path_temp.append(path_length(d,balanced_path_list,size))
        best_lens[t] = best_path
    avg_p = best_lens.mean()
    std_p = best_lens.std()
    best_p = best_lens.min()
    worst_p = best_lens.max()
    print('20次测试：')
    print('最佳性能：%.6f' % best_p)
    print('最差性能：%.6f' % worst_p)
    print('平均性能：%.6f' % avg_p)
    print('性能标准差：%.6f' % std_p)