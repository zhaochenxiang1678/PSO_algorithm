import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class Particle:
    ## 粒子类
    def __init__(self,dim,bounds):
        self.dim=dim
        self.bounds=bounds
        """粒子类特征：
        位置，速度，适应度，自身最优位置和适应度
        """
        self.position=np.random.uniform(bounds[0],bounds[1],dim)
        self.velocity=np.random.uniform(-1,1,dim)
        self.fitness=float('inf')
        self.best_position=self.position.copy()
        self.best_fitness=float('inf')

class PSO:
    def __init__(self,n_particles,dim,bounds,fitness_func,
                 w=0.7,c1=2.0,c2=2.0,max_iter=100,tolerance=1e-6):
        """
        初始化PSO算法

        参数:
        n_particles: 粒子数量
        dim: 问题维度
        bounds: 搜索空间边界 [lower_bound, upper_bound]
        fitness_func: 适应度函数
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        max_iter: 最大迭代次数
        tolerance: 收敛容差
        """
        self.n_particles=n_particles
        self.dim=dim
        self.bounds=bounds
        self.fitness_func=fitness_func
        self.w=w
        self.c1=c1
        self.c2=c2
        self.max_iter=max_iter
        self.tolerance=tolerance

        # 记录全局最优位置和适应度
        self.global_best_position=None
        self.global_best_fitness=float('inf')

        # 初始化粒子群
        self.particles=[Particle(dim,bounds) for _ in range(n_particles)]

        # 记录迭代过程，便于绘图
        self.history={
            'best_fitness':[],
            'avg_fitness':[],
            'iteration':[]
        }

        # 收敛记号
        self.converged=False

    def evaluate_fitness(self,particle):
        """评估粒子适应度"""
        try:
            particle.fitness=self.fitness_func(particle.position)
        except:
            particle.fitness=float('inf')

        # 更新个体最优
        if particle.fitness<particle.best_fitness:
            particle.best_position=particle.position.copy()
            particle.best_fitness=particle.fitness
            # 更新全局最优
            if particle.fitness<self.global_best_fitness:
                self.global_best_position=particle.position.copy()
                self.global_best_fitness=particle.fitness

    def update_velocity(self,particle):
        """更新粒子速度"""
        r1,r2=np.random.rand(2)
        cognitive_velocity=self.c1*r1*(particle.best_position-particle.position)
        social_velocity=self.c2*r2*(self.global_best_position-particle.position)
        particle.velocity=self.w*particle.velocity+cognitive_velocity+social_velocity

        # 速度限制
        max_velocity=(self.bounds[1]-self.bounds[0])*0.1
        particle.velocity=np.clip(particle.velocity,-max_velocity,max_velocity)

    def update_position(self,particle):
        """更新粒子位置"""
        particle.position+=particle.velocity

        # 位置限制
        particle.position=np.clip(particle.position,self.bounds[0],self.bounds[1])

    def check_convergence(self):
        """检查是否收敛"""
        if len(self.history['best_fitness'])<10:
            return False

        recent_fitness=self.history['best_fitness'][-10:]
        improvement=abs(recent_fitness[0]-recent_fitness[-1])

        return improvement<self.tolerance

    def optimize(self):
        """执行优化"""
        print(f'开始PSO优化')
        print('-'*50)
        print(f'粒子数量：{self.n_particles}')
        print(f'粒子维度：{self.dim}')
        print(f'搜索空间约束：{self.bounds[0]},{self.bounds[1]}')
        print(f'最大迭代：{self.max_iter}')
        print('-'*50)

        # 记录迭代起始时间
        start_time=time.time()
        for iteration in range(self.max_iter):
            # 评估所有粒子
            for particle in self.particles:
                self.evaluate_fitness(particle)
            # 评估完毕所有粒子之后，得到全部最优
            # 接着更新速度和位置
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)

            # 记录历史
            avg_fitness=np.mean([p.fitness for p in self.particles])
            self.history['best_fitness'].append(self.global_best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['iteration'].append(iteration+1)

            # 检查收敛
            if self.check_convergence():
                self.converged=True
                print(f'算法在第{iteration+1}次迭代后收敛')
                break

            # 打印进度
            if (iteration+1)%10==0:
                print(f'迭代[{iteration+1}/{self.max_iter}]:',
                f'最优质={self.global_best_fitness:.8f}',
                f'平均={avg_fitness:.8f}')

        # 迭代结束时间
        end_time=time.time()

        print("-"*50)
        print(f'优化完成！')
        print(f'最优解：{self.global_best_position}')
        print(f'最优适应度：{self.global_best_fitness}')
        print(f'收敛状态：{'是' if self.converged else '否'}')
        print(f'总耗时：{end_time-start_time:.3f}秒')
        print(f'总迭代：{len(self.history['iteration'])}')

        return self.global_best_position,self.global_best_fitness

    def plot_convergence(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.history['iteration'],self.history['best_fitness'],
                 'b-',linewidth=2,label='全局最优')
        plt.plot(self.history['iteration'],self.history['avg_fitness'],
                 'r--',linewidth=1,label='平均适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.title('PSO收敛曲线')
        plt.legend()
        plt.grid()
        plt.show()

    def get_statistics(self):
        """获取优化统计信息"""
        if not self.history['best_fitness']:
            return {}

        states={
            'final_best_fitness':self.history['best_fitness'][-1],
            'final_avg_fitness':self.history['avg_fitness'][-1],
            'total_iterations':len(self.history['iteration']),
            'converged':self.converged,
            'improvement':self.history['best_fitness'][0]-self.history['best_fitness'][-1]
        }
        return states

    @staticmethod
    def sphere(x):
        """Sphere函数f(x)=x^2(求和)"""
        return np.sum(x**2)

if __name__=='__main__':
    np.random.seed(42)

    # 测试参数
    n_particles=30
    dim=2
    bounds=[-10,10]
    max_iter=1000

    print("=== PSO算法测试 ===")

    print("Sphere函数测试")
    pso=PSO(n_particles=n_particles,dim=dim,bounds=bounds,
            fitness_func=PSO.sphere,max_iter=max_iter)

    best_position,best_fitness=pso.optimize()
    pso.plot_convergence()

    # 显示统计信息
    print('-'*50)
    print('统计信息：')
    states=pso.get_statistics()
    print(states)
