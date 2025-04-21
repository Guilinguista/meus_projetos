# K-Means Clustering from Scratch – 2D Points

import random
import math
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

# Generate random 2D points
random.seed(42)
n_points = 1000
points = [(random.random(), random.random()) for _ in range(n_points)]

# Number of clusters
k = 3

# Initialize centroids randomly
centroids = random.sample(points, k)

# K-means loop
max_iterations = 100
tolerance = 1e-4

for iteration in range(max_iterations):
    # Assignment step
    clusters = [[] for _ in range(k)]
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)

    # Update step
    new_centroids = []
    for cluster in clusters:
        if cluster:
            x_coords = [p[0] for p in cluster]
            y_coords = [p[1] for p in cluster]
            new_centroids.append((sum(x_coords) / len(cluster), sum(y_coords) / len(cluster)))
        else:
            new_centroids.append(random.choice(points))

    # Check for convergence
    shift = sum(euclidean_distance(c, nc) for c, nc in zip(centroids, new_centroids))
    if shift < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
    centroids = new_centroids
else:
    print("Reached max iterations.")

# Plotting result
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, cluster in enumerate(clusters):
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    plt.scatter(xs, ys, color=colors[i % len(colors)], alpha=0.6, label=f"Cluster {i+1}")

cx = [c[0] for c in centroids]
cy = [c[1] for c in centroids]
plt.scatter(cx, cy, color='black', marker='x', s=100, label='Centroids')

plt.title("K-Means Clustering (from Scratch)")
plt.legend()
plt.grid(True)
plt.show()
