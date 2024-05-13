from math import fsum, sqrt
from typing import Tuple, Iterable, Sequence, Dict, List
from collections import defaultdict
from functools import partial
from random import sample


Point = Tuple[int, ...]
Centroid = Point

def transpose(data: Iterable[Sequence[float]]) -> List[Tuple[float, ...]]:
    'Swap rows and columns in a 2-D array of data'
    return list(zip(*data))

def mean(data: Iterable[float]) -> float:
    'Accurate arithmetic mean'
    data = list(data)
    return fsum(data) / len(data)

def dist(p: Point, q: Point) -> float:
    'Euclidean distance function for multi-dimensional data'
    return sqrt(fsum((x - y)**2 for x, y in zip(p, q)))

def assign_data(centroids: Sequence[Point], data: Iterable[Point]) -> Dict[Centroid, List[Point]]:
    'Group data points by the closest centroid'
    d = defaultdict(list)
    for point in data:
        closest_centroid = min(centroids, key=partial(dist, point))
        d[closest_centroid].append(point)
    return dict(d)

def compute_centroids(groups: Iterable[Sequence[Point]]) -> List[Centroid]:
    'Compute the centroid of each group'
    return [tuple(map(mean, transpose(group))) for group in groups]

def k_means(data: Iterable[Point], k: int = 3, iterations: int = 50) -> List[Centroid]:
    'Perform k-means clustering'
    data = list(data)
    if not data:
        raise ValueError("Data should not be empty")
    if k > len(data):
        raise ValueError("Number of clusters cannot exceed number of data points")
    
    centroids = sample(data, k)

    for _ in range(iterations):
        labeled = assign_data(centroids, data)
        centroids = compute_centroids(labeled.values())
    return centroids
