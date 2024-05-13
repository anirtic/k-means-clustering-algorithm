# k-means-clustering-algorithm

This repository contains a manually implemented version of the k-means clustering algorithm in Python. It is intended as a learning project to understand the fundamentals of clustering algorithms and unsupervised machine learning.

## Overview

k-means clustering is an unsupervised learning algorithm partitioning a dataset into `k` distinct, non-overlapping subsets (clusters). The algorithm aims to minimize the variance within each cluster, making the clusters as distinct as possible.

## Features

- Manual implementation of k-means clustering
- Euclidean distance calculation
- Centroid computation
- Grouping data points by the closest centroid
- Iterative refinement of centroids

## Usage

To use the k-means clustering algorithm, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/k-means-clustering-algorithm.git
    ```

2. Navigate to the project directory:
    ```bash
    cd k-means-clustering-algorithm
    ```

3. Run the k-means algorithm with your data:
    ```python
    from kmeans import k_means

    data = [
        (1, 2), (2, 3), (3, 4), (8, 9), (9, 10), (10, 11)
    ]
    k = 2
    centroids = k_means(data, k)

    print("Computed centroids:", centroids)
    ```

## Functions

- **transpose(data)**: Swaps rows and columns in a 2-D array of data.
- **mean(data)**: Calculates the arithmetic mean of a list of numbers.
- **dist(p, q)**: Computes the Euclidean distance between two multi-dimensional points.
- **assign_data(centroids, data)**: Groups data points by the closest centroid.
- **compute_centroids(groups)**: Computes the centroid of each group.
- **k_means(data, k=3, iterations=50)**: Performs k-means clustering on the given data.
