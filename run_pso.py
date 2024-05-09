import time
import psutil
import numpy as np
from pso_optim import PSO
import matplotlib.pyplot as plt


def objective_function(x):
    return -np.sin(x[0]) * np.cos(x[1])


def plot_fig(lb, ub, best_position, best_fitness):

    # Create a grid of points in the search space
    x = np.linspace(lb[0], ub[0], 100)
    y = np.linspace(lb[1], ub[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Evaluate the objective function at each point in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function([X[i, j], Y[i, j]])

    # Plot the objective function surface and the best solution found by PSO
    fig = plt.figure()
    ax2 = fig.add_subplot(projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
    ax2.scatter(best_position[0], best_position[1],
                best_fitness, c='r', s=100, marker='x')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Objective Function')
    ax2.set_title('PSO Optimization of 2D Function')
    plt.show()


def main():

    # Define the search space bounds
    lb = [-5, -5]
    ub = [5, 5]

    pso = PSO(objective_function, lb, ub,
              num_particles=1200, num_iterations=50)

    start_mem = psutil.virtual_memory().used
    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()
    best_position, best_fitness, fitness_history = pso.optimize()
    print("Serial Execution Time:", time.time() - start_time)
    print("Best Fitness:", best_fitness)
    end_cpu = psutil.cpu_percent(interval=None)
    print("CPU Usage:", end_cpu - start_cpu)
    print("Memory Usage GB:", (psutil.virtual_memory().used - start_mem) / (1024 ** 3))

    start_mem = psutil.virtual_memory().used
    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()
    best_position, best_fitness = pso.optimize_fast()
    print("Parallel Execution Time:", time.time() - start_time)
    print("Best Fitness:", best_fitness)
    end_cpu = psutil.cpu_percent(interval=None)
    print("CPU Usage:", end_cpu - start_cpu)
    print("Memory Usage GB:", (psutil.virtual_memory().used - start_mem) / (1024 ** 3))

    start_mem = psutil.virtual_memory().used
    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()
    best_position, best_fitness = pso.optimize_async()
    print("Async Execution Time:", time.time() - start_time)
    print("Best Fitness:", best_fitness)
    end_cpu = psutil.cpu_percent(interval=None)
    print("CPU Usage:", end_cpu - start_cpu)
    print("Memory Usage GB:", (psutil.virtual_memory().used - start_mem) / (1024 ** 3))

    # plot_fig(lb, ub, best_position, best_fitness)


if __name__ == '__main__':
    main()
