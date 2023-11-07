import numpy as np
import random

distance_matrix = np.array([
    [0, 29, 20, 21, 16],
    [29, 0, 15, 17, 28],
    [20, 15, 0, 28, 18],
    [21, 17, 28, 0, 12],
    [16, 28, 18, 12, 0]
])

num_cities = len(distance_matrix)

def nearest_neighbor_initialization():
    start_city = random.randint(0, num_cities - 1)
    tour = [start_city]
    remaining_cities = list(range(num_cities))
    remaining_cities.remove(start_city)

    while remaining_cities:
        current_city = tour[-1]
        nearest_city = min(remaining_cities, key=lambda city: distance_matrix[current_city][city])
        tour.append(nearest_city)
        remaining_cities.remove(nearest_city)

    return tour

def shortest_edge_initialization():
    tour = [0]
    remaining_cities = list(range(1, num_cities))

    while remaining_cities:
        current_city = tour[-1]
        next_city = min(remaining_cities, key=lambda city: distance_matrix[current_city][city])
        tour.append(next_city)
        remaining_cities.remove(next_city)

    tour.append(tour[0]) 
    return tour

def partially_matched_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)
    child1 = [None] * n
    child2 = [None] * n

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    for i in range(n):
        if child1[i] is None:
            city = parent2[i]
            while city in child1:
                index = parent2.index(city)
                city = parent2[(index + 1) % n]
            child1[i] = city

        if child2[i] is None:
            city = parent1[i]
            while city in child2:
                index = parent1.index(city)
                city = parent1[(index + 1) % n]
            child2[i] = city

    return child1, child2

def order_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)
    child1 = [-1] * n
    child2 = [-1] * n

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    j1 = end
    j2 = end

    for i in range(n):
        if parent2[(i + end) % n] not in child1:
            child1[j1 % n] = parent2[(i + end) % n]
            j1 += 1

        if parent1[(i + end) % n] not in child2:
            child2[j2 % n] = parent1[(i + end) % n]
            j2 += 1

    return child1, child2

def mutation_by_inversion(tour):
    start = random.randint(0, len(tour) - 1)
    end = random.randint(start + 1, len(tour))
    tour[start:end] = reversed(tour[start:end])
    return tour

def mutation_by_insertion(tour):
    city1, city2 = random.sample(tour, 2)
    index1 = tour.index(city1)
    index2 = tour.index(city2)
    tour.insert(index1, city2)
    tour.pop(index2 + 1)
    return tour

print("Graph generation and display:")
print("Distance Matrix:")
print(distance_matrix)

nn_tour = nearest_neighbor_initialization()
print("\nNearest neighbor initialization:")
print(nn_tour)
print(nn_tour + [nn_tour[0]])  

se_tour = shortest_edge_initialization()
print("\nShortest edge initialization:")
print(se_tour)
print(se_tour + [se_tour[0]]) 

parent1 = [0, 1, 2, 3, 4]
parent2 = [4, 3, 2, 1, 0]
print("\nPartially matched crossover:")
print(partially_matched_crossover(parent1, parent2))

print("\nOrder crossover:")
print(order_crossover(parent1, parent2))

tour = [0, 1, 2, 3, 4]
print("\nMutation by inversion:")
print(mutation_by_inversion(tour))

print("\nMutation by insertion:")
print(mutation_by_insertion(tour))







import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
num_nodes = distance_matrix.shape[0]
G.add_nodes_from(range(num_nodes))

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        G.add_edge(i, j, weight=distance_matrix[i, j])

pos = nx.circular_layout(G)

nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=10, font_color='black')
edge_labels = {(i, j): f'{distance:.2f}' for (i, j, distance) in G.edges(data='weight')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Distance Matrix Graph")
plt.axis('off')
plt.show()
