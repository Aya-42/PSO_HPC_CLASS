import numpy as np
from multiprocessing import Pool, Manager, Lock
from typing import Callable, List, Tuple


class Particle:
    """
    Represents a particle in the Particle Swarm Optimization algorithm.
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, w: float = 0.5, c1: float = 2.05, c2: float = 2.05):
        """
        Initialize a particle with random position and zero velocity within the given bounds.

        Parameters:
        lb (np.ndarray): The lower bounds of the search space.
        ub (np.ndarray): The upper bounds of the search space.
        w (float): Inertia weight factor.
        c1 (float): Cognitive (personal) coefficient.
        c2 (float): Social (global) coefficient.
        """
        if lb.shape != ub.shape:
            raise ValueError(
                "Lower and upper bounds must have the same dimensions")
        self.position = np.random.uniform(lb, ub)
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_fitness = np.inf
        self.fitness = np.inf
        self.lb = lb
        self.ub = ub
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_state(self, global_best_position: np.ndarray, lb: np.ndarray, ub: np.ndarray, w: float, c1: float, c2: float):
        """
        Update the state of the particle (position and velocity) based on its own experience
        and that of the global best.

        Parameters:
        global_best_position (np.ndarray): The best position found by any particle so far.
        lb (np.ndarray): The lower bounds of the search space.
        ub (np.ndarray): The upper bounds of the search space.
        w (float): Inertia weight factor.
        c1 (float): Cognitive (personal) coefficient.
        c2 (float): Social (global) coefficient.
        """
        r1 = np.random.uniform(0, 1, size=len(self.position))
        r2 = np.random.uniform(0, 1, size=len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.position += self.velocity
        self.position = np.clip(self.position, lb, ub)

    def update_fitness(self, objective_function: Callable[[np.ndarray], float]):
        """
        Update the fitness of the particle and adjust the best position if the current
        position is better.

        Parameters:
        objective_function (Callable): The function to calculate the fitness of a position.
        """
        self.fitness = objective_function(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = np.copy(self.position)
        return self


def update_particle_fitness(particle: Particle, objective_function: Callable[[np.ndarray], float]) -> Particle:
    """
    Update the fitness of a single particle.

    Parameters:
    particle (Particle): The particle to update.
    objective_function (Callable): The fitness function to apply.
    """
    return particle.update_fitness(objective_function)


def evaluate_and_update(particle, objective_function, global_best, lock):
    current_fitness = objective_function(particle.position)
    with lock:
        # Check and update global best if needed
        if current_fitness < global_best['fitness']:
            global_best['fitness'] = current_fitness
            global_best['position'] = particle.position.copy()

# hello


class PSO:
    """
    A class to represent the Particle Swarm Optimization algorithm.
    """

    def __init__(self, objective_function: Callable[[np.ndarray], float], lb: List[float], ub: List[float], num_particles: int = 50, w: float = 0.5, c1: float = 2.05, c2: float = 2.05, num_iterations: int = 100):
        """
        Initialize the PSO algorithm with the given parameters.

        Parameters:
        objective_function (Callable): The function to calculate the fitness of a position.
        lb (List[float]): The lower bounds of the search space.
        ub (List[float]): The upper bounds of the search space.
        num_particles (int): The number of particles in the swarm.
        w (float): Inertia weight factor.
        c1 (float): Cognitive (personal) coefficient.
        c2 (float): Social (global) coefficient.
        num_iterations (int): The number of iterations to perform.
        """
        if len(lb) != len(ub):
            raise ValueError(
                "Lower and upper bounds lists must be of the same length")
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        self.objective_function = objective_function
        self.particles = [Particle(self.lb, self.ub)
                          for _ in range(num_particles)]
        self.gbest_position = None
        self.gbest_fitness = np.inf

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the PSO algorithm without parallel processing.

        Returns:
        Tuple[np.ndarray, float]: The global best position and its corresponding fitness.
        """
        best_fitness_history = []  # Store the history of best fitness values

        for _ in range(self.num_iterations):
            for particle in self.particles:
                particle.update_fitness(self.objective_function)

                # Update the global best
                if particle.best_fitne`ss < self.gbest_fitness:
                    self.gbest_fitness = particle.best_fitness
                    self.gbest_position = np.copy(particle.best_position)

            best_fitness_history.append(self.gbest_fitness)

            for particle in self.particles:
                particle.update_state(
                    self.gbest_position, self.lb, self.ub, self.w, self.c1, self.c2)

        return self.gbest_position, self.gbest_fitness, best_fitness_history

    def optimize_fast(self) -> Tuple[np.ndarray, float]:
        """
        Run the PSO algorithm with parallel processing to potentially improve performance.

        Returns:
        Tuple[np.ndarray, float]: The global best position and its corresponding fitness.
        """
        with Pool() as pool:
            for _ in range(self.num_iterations):
                # Parallel evaluation of fitness using a defined function
                self.particles = pool.starmap(update_particle_fitness, [(
                    p, self.objective_function) for p in self.particles])

                # Update global best
                for particle in self.particles:
                    if particle.best_fitness < self.gbest_fitness:
                        self.gbest_fitness = particle.best_fitness
                        self.gbest_position = np.copy(particle.best_position)

                # Update particles' states
                for particle in self.particles:
                    particle.update_state(
                        self.gbest_position, self.lb, self.ub, self.w, self.c1, self.c2)

        return self.gbest_position, self.gbest_fitness

    def optimize_async(self) -> Tuple[np.ndarray, float]:
        with Manager() as manager:
            global_best = manager.dict(fitness=np.inf, position=None)
            lock = manager.Lock()

            with Pool() as pool:
                for _ in range(self.num_iterations):
                    # Parallel evaluation and potential update
                    pool.starmap(evaluate_and_update, [
                                (p, self.objective_function, global_best, lock) for p in self.particles])

                    # Synchronize particles' states after updating the global best
                    for particle in self.particles:
                        particle.update_state(
                            global_best['position'], particle.lb, particle.ub, particle.w, particle.c1, particle.c2)
            return global_best['position'], global_best['fitness']
