# Particle Swarm Optimization

  ## Installation
```
git clone https://github.com/Aya-42/PSO_HPC_CLASS
```

## Usage

Currently the package provides the `mopso.py` module that defines two classes: `MOPSO` and `Particle`. To use this module in your Python projects:

1. Import the required modules:
   
   ```python
   from optimizer import MOPSO
   ```

2. Define the objective function to be optimized.
   
   ```python
   def objective_function(x):
       return np.sin(x[0]) + np.sin(x[1])
   ```

3. Create the MOPSO object with the configuration of the algorithm
   
   ```python
   mopso = PSO(
       objective_functions=[objective_function],
       lower_bound=-5,
       upper_bound=5,
       num_particles=50,
       num_iterations=100
   )
   ```

4. Run the optimization algorithm
   
   ```python
   ughhhhh= mopso.optimize()
   ```

See [run_mopso.py](example/run_mopso.py) for an example.

## Tests
