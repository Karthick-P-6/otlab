import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2 

class Particle:
    def __init__(self, num_dimensions, lb, ub):
        self.position = np.random.uniform(lb, ub, num_dimensions)
        self.velocity = np.random.uniform(-1, 1, num_dimensions)
        self.best_position = self.position
        self.best_fitness = float('inf')

def particle_swarm_optimization(objective_function, num_particles, num_dimensions, lb, ub, num_iterations, w, c1, c2):
    particles = [Particle(num_dimensions, lb, ub) for _ in range(num_particles)]
    global_best_position = np.zeros(num_dimensions)
    global_best_fitness = float('inf')

    for iteration in range(num_iterations):
        for particle in particles:

            fitness = objective_function(particle.position)

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            new_velocity = (w * particle.velocity +
                            c1 * r1 * (particle.best_position - particle.position) +
                            c2 * r2 * (global_best_position - particle.position))
            particle.velocity = new_velocity
            particle.position += particle.velocity

    return global_best_position, global_best_fitness

num_dimensions = 2
lower_bound = np.array([-5, -5])
upper_bound = np.array([5, 5])

num_particles = 50
num_iterations = 10
w = 0.729 
c1 = 1.49445  
c2 = 1.49445  

best_solution, best_fitness = particle_swarm_optimization(objective_function, num_particles, num_dimensions, lower_bound, upper_bound, num_iterations, w, c1, c2)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
