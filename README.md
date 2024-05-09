# Particle Swarm Optimization
PSO is a global optimization technique inspired by the social behavior of bird flocking or fish schooling. Tt requires only the objective function and does not depend on the gradient or any differential form of the objective.  It also has very few hyperparameters.

  ## Installation
```
git clone https://github.com/Aya-42/PSO_HPC_CLASS
```

## Usage

Currently the package provides the `pso.py` module that defines two classes: `PSO` and `Particle`. To use this module in your Python projects:

1. Import the required modules:
   
   ```python
   from pso_optim import PSO
   ```

2. Define the objective function to be optimized.
   
   ```python
   def objective_function(x):
       return np.sin(x[0]) + np.sin(x[1])
   ```

3. Create the PSO object with the configuration of the algorithm
   
   ```python
   pso = PSO(
       objective_function,
       lower_bound=[-5,-5],
       upper_bound=[5,5],
       num_particles=50,
       num_iterations=100
   )
   ```

4. Run the optimization algorithm
   
   ```python
       best_position, best_fitness, fitness_history = pso.optimize()= pso.optimize()
   ```

See [run_pso.py](run_pso.py) for an example.
