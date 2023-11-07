import numpy as np

def objective_function(x):
    return sum(x**2)

def gray_wolf_optimization(objective_function, num_wolves, num_iterations, lb, ub):
    num_dimensions = len(lb)
    wolves_position = np.random.uniform(lb, ub, (num_wolves, num_dimensions))
    alpha_position = np.zeros(num_dimensions)
    beta_position = np.zeros(num_dimensions)
    delta_position = np.zeros(num_dimensions)
    alpha_score = float('inf')
    beta_score = float('inf')
    delta_score = float('inf')

    for iteration in range(num_iterations):
        for i in range(num_wolves):
            fitness = objective_function(wolves_position[i])
            if fitness < alpha_score:
                delta_score = beta_score
                delta_position = beta_position.copy()
                beta_score = alpha_score
                beta_position = alpha_position.copy()
                alpha_score = fitness
                alpha_position = wolves_position[i].copy()
            elif alpha_score < fitness < beta_score:
                delta_score = beta_score
                delta_position = beta_position.copy()
                beta_score = fitness
                beta_position = wolves_position[i].copy()
            elif alpha_score < fitness < delta_score:
                delta_score = fitness
                delta_position = wolves_position[i].copy()

        a = 2 - iteration * (2 / num_iterations) 

        for i in range(num_wolves):
            r1 = np.random.random(num_dimensions)
            r2 = np.random.random(num_dimensions)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * alpha_position - wolves_position[i])
            X1 = alpha_position - A1 * D_alpha

            r1 = np.random.random(num_dimensions)
            r2 = np.random.random(num_dimensions)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * beta_position - wolves_position[i])
            X2 = beta_position - A2 * D_beta

            r1 = np.random.random(num_dimensions)
            r2 = np.random.random(num_dimensions)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * delta_position - wolves_position[i])
            X3 = delta_position - A3 * D_delta

            wolves_position[i] = (X1 + X2 + X3) / 3

    return alpha_position, alpha_score

num_dimensions = 3
lower_bound = np.array([-10, -10])
upper_bound = np.array([10, 10])

num_wolves = 10
num_iterations = 10

best_solution, best_fitness = gray_wolf_optimization(objective_function, num_wolves, num_iterations, lower_bound, upper_bound)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
