import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

class SGAOptimizer(object):

    def __init__(self, decision_lb, decision_ub, precision, objective_fn, target, init_pop_size, prob_mutation, prob_crossover):
        
        self._lb = decision_lb
        self._ub = decision_ub
        self._prec = precision
        self._obj_fn = objective_fn
        self._code_length = np.ceil(np.log2(((decision_ub - decision_lb) / precision))).astype(np.uint8)
        self._interval = (decision_ub - decision_lb) / (1 << self._code_length)
        self._pop_size = init_pop_size
        self._pm = prob_mutation
        self._pc = prob_crossover
        self._target = target

        self._population = np.round(np.random.random([self._pop_size, self._code_length])).astype(np.uint8)
        self._base_vec = np.array([[1 << x for x in reversed(range(self._code_length))]])
        self._fit_value = np.zeros([self._pop_size, 1])
        self._best_sample = self._population[0,:]

    def reset(self):

        self._population = np.round(np.random.random([self._pop_size, self._code_length])).astype(np.uint8)
        self._fit_value = np.zeros([self._pop_size, 1])
        self._best_sample = self._population[0,:]

    def _select(self):
        
        if self._target == 'min':
            self._fit_value = self._fit_value.max() - self._fit_value

        cpf = np.cumsum(self._fit_value / self._fit_value.sum())
        new_pop = np.zeros_like(self._population)
        basket = np.sort(np.random.random(self._pop_size))
        
        newin = 0
        fitin = 0
        
        while newin < self._pop_size:
            if basket[newin] < cpf[fitin]:
                new_pop[newin,:] = self._population[fitin,:]
                newin += 1
            else:
                fitin += 1
        
        self._population = new_pop

    def _mutate(self):
        
        for i in range(self._pop_size):
            if np.random.random() < self._pm:
                mutate_pt = np.floor(np.random.random() * self._code_length).astype(np.uint8)
                self._population[i,mutate_pt] = 1 - self._population[i,mutate_pt]

    def _crossover(self):
        
        for i in range(0, self._pop_size, 2):
            if np.random.random() < self._pc:
                cross_pt = np.floor(np.random.random() * self._code_length).astype(np.uint8)
                tmp = self._population[i + 1, cross_pt:]
                self._population[i + 1, cross_pt:] = self._population[i, cross_pt:]
                self._population[i, cross_pt:] = tmp
    
    def _eval_fitness(self):
        
        self._fit_value = self._obj_fn(self._decode())
    
    def _decode(self):
        
        base2 = np.repeat(self._base_vec, self._pop_size, 0)
        solution = np.sum(self._population * base2, axis=1, keepdims=True)
        solution = solution * self._interval + self._lb
        
        return solution

    def step(self):
        
        self._eval_fitness()
        self._select()
        self._crossover()
        self._mutate()
    
    def eval(self):
        
        solution = self._decode()
        fit_vals = self._obj_fn(solution)

        return solution, fit_vals

def main():

    # Griewank 1D Function
    def Griewank1D(x):
        return 1 + x ** 2 / 4000 - np.cos(x)
    
    # SGA
    lb = -600
    ub = 600
    prec = 1e-4
    obj_fn = Griewank1D
    init_pop_size = 1000
    pm = 0.01
    pc = 0.01

    optimizer = SGAOptimizer(lb, ub, prec, obj_fn, 'min', init_pop_size, pm, pc)

    # Iterations
    steps = 100
    best_Y = np.zeros([steps])
    mean_Y = np.zeros([steps])
    ts = np.linspace(0, steps, steps)
    for t in range(steps):
        optimizer.step()
        _, Ys = optimizer.eval()
        best_Y[t] = Ys.min()
        mean_Y[t] = Ys.mean()
    
    result = best_Y.min()
    print('最优值为：%.6f' % result)

    sns.set()
    plt.plot(ts, best_Y)
    plt.plot(ts, mean_Y)
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    plt.legend(['个体最优值', '种群平均值'])
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.title('目标函数值变化曲线')
    plt.show()

    # Stats
    best_Ys = np.zeros([20])
    for T in range(20):
        optimizer.reset()
        for t in range(steps):
            optimizer.step()
        _, Ys = optimizer.eval()
        best_Ys[T] = Ys.min()
    
    best_perfomance = best_Ys.min()
    worst_perfomance = best_Ys.max()
    avg_perfomance = best_Ys.mean()
    std_perfomance = best_Ys.std()

    print('20次实验：')
    print('最佳性能：%.6f' % best_perfomance)
    print('最差性能：%.6f' % worst_perfomance)
    print('平均性能：%.6f' % avg_perfomance)
    print('性能标准差：%.6f' % std_perfomance)

if __name__ == '__main__':
   main()
