import numpy as np
import unittest
from PSO import Particle, PSO


def objective_function(x):
    return np.sum(x ** 2)


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.lb = np.array([-5, -5])
        self.ub = np.array([5, 5])
        self.particle = Particle(self.lb, self.ub)

    def test_particle_initialization(self):
        self.assertTrue(np.all(self.lb <= self.particle.position))
        self.assertTrue(np.all(self.particle.position <= self.ub))
        self.assertTrue(np.all(self.particle.velocity == 0))
        self.assertTrue(
            np.all(self.particle.best_position == self.particle.position))
        self.assertEqual(self.particle.best_fitness, np.inf)
        self.assertEqual(self.particle.fitness, np.inf)

    def test_update_fitness(self):
        self.particle.update_fitness(objective_function)
        self.assertNotEqual(self.particle.fitness, np.inf)
        self.assertTrue(self.particle.best_fitness <= self.particle.fitness)
        self.assertTrue(
            np.all(self.particle.best_position == self.particle.position))


class TestPSO(unittest.TestCase):
    def setUp(self):
        self.lb = [-5, -5]
        self.ub = [5, 5]
        self.num_particles = 50
        self.num_iterations = 100
        self.pso = PSO(objective_function, self.lb, self.ub,
                       self.num_particles, num_iterations=self.num_iterations)

    def test_pso_initialization(self):
        self.assertEqual(len(self.pso.particles), self.num_particles)
        self.assertEqual(self.pso.w, 0.5)
        self.assertEqual(self.pso.c1, 2.05)
        self.assertEqual(self.pso.c2, 2.05)
        self.assertEqual(self.pso.num_iterations, self.num_iterations)

    def test_optimize(self):
        gbest_position, gbest_fitness = self.pso.optimize()
        self.assertNotEqual(gbest_fitness, np.inf)
        self.assertTrue(np.all(self.lb <= gbest_position))
        self.assertTrue(np.all(gbest_position <= self.ub))

    def test_optimize_fast(self):
        gbest_position, gbest_fitness = self.pso.optimize_fast()
        self.assertNotEqual(gbest_fitness, np.inf)
        self.assertTrue(np.all(self.lb <= gbest_position))
        self.assertTrue(np.all(gbest_position <= self.ub))


if __name__ == '__main__':
    unittest.main()
