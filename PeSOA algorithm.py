import numpy as np
import random
import matplotlib.pyplot as plt

class PenguinSearchOptimizer:
    def __init__(self, num_penguins, dimensions, bounds):
        self.num_penguins = num_penguins
        self.dimensions = dimensions
        self.bounds = bounds
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_penguins, dimensions))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(num_penguins, float('inf'))

    def fitness_function(self, x):
        cost = np.sum(x)
        displacement = np.sum(1 / (x + 1e-6)) * 0.01  
        stress = np.max(x) * 25e6                    

        penalty = 0
        if displacement > 0.05:
            penalty += 1e7  

        if stress > 250e6:
            penalty += 1e7

        return cost + penalty

    def optimize(self, max_iterations):
        global_best_position = self.positions[0].copy()
        global_best_fitness = float('inf')
        convergence_history = [] 

        for iteration in range(max_iterations):
            for i in range(self.num_penguins):
                current_fitness = self.fitness_function(self.positions[i])
                
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.positions[i].copy()

                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = self.positions[i].copy()

            convergence_history.append(global_best_fitness)

            for i in range(self.num_penguins):
                r1, r2 = random.random(), random.random()
                cognitive_component = 2.0 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = 2.0 * r2 * (global_best_position - self.positions[i])
                self.positions[i] += cognitive_component + social_component
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

        return global_best_position, global_best_fitness, convergence_history

NUM_PENGUINS = 100
DIMENSIONS = 12   
BOUNDS = (0.1, 10.0)
MAX_ITERATIONS = 100

optimizer = PenguinSearchOptimizer(NUM_PENGUINS, DIMENSIONS, BOUNDS)
best_pos, best_score, convergence_data = optimizer.optimize(MAX_ITERATIONS)

print(f"Optimization Completed.")
print(f"Minimum Cost (Best Fitness): {best_score:.4f}")
print(f"Optimal Design Variables: \n{best_pos}")

plt.figure(figsize=(10, 6))
plt.plot(convergence_data, linewidth=2, color='#1f77b4', label='Global Best (Cost + Penalty)')
plt.title('Penguin Search Algorithm Convergence Curve')
plt.xlabel('Iteration Number')
plt.ylabel('Fitness Value')
plt.yscale('log') 
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()