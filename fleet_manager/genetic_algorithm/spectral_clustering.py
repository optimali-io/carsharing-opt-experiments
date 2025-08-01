from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

from fleet_manager.data_model.genetic_model import Phenotype


def split_clusters(limit: int, clusters: List[List[Phenotype]]) -> List[List[Phenotype]]:
    """
    Function splits clusters of Phenotypes into cluster of size smaller than limit.

    :param limit: maximum size of the cluster
    :param clusters: list of lists of Phenotypes
    """
    new_clusters = []
    for c in clusters:
        if len(c) >= 2 * limit:
            new_clusters.extend([c[x : x + limit] for x in range(0, len(c), limit)])
        else:
            new_clusters.append(c)
    return new_clusters


# For now there is only similarity function (others proved useless), but in the future we may implement more
# sophisticated functions, so I think that interface should stay anyway.
class ISpecimenSimilarity:
    """
    Interface class for implementing similarity functions between phenotypes using k-neighbors algorithm.

    :param n_neighbors: Number of neighbors for each sample for k-neighbors algorithm.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors: int = n_neighbors

    def get_adjacency_matrix(self, population: List[Phenotype]) -> np.ndarray:
        """
        Abstract function to implement in derived class, that should returning matrix of similarities.

        :param population: list of Phenotypes

        :return: NxN matrix of similarities
        """

        raise NotImplementedError


class RevenueCostChromosomeLenSimilarity(ISpecimenSimilarity):
    """
    Implementation of the `ISpecimenSimilarity` interface, building similarities using revenue, cost, and number of
    genes.
    """

    def get_adjacency_matrix(self, population: List[Phenotype]) -> np.ndarray:
        """
        For given list of Phenotypes of size N, create NxN matrix with similarity values
        (k-neighbors graph edge weights) between Phenotypes.

        :param population: list of Phenotypes

        :return: NxN matrix of similarities
        """
        sim_attr_array = np.array([(p.diff_revenue, p.cost, p.genome.gene_counter) for p in population])
        return kneighbors_graph(sim_attr_array, n_neighbors=self.n_neighbors).toarray()


class SpectralClustering:
    """Class responsible for creating clusters of similar Phenotypes."""

    def __init__(
        self, similarity_attributes_creator: ISpecimenSimilarity, max_island_size: int, tolerance: float = 0.1
    ):
        """
        :param similarity_attributes_creator: Instance extending ISpecimenSimilarity interface
        :param tolerance: how close to zero must eigenvalues of adjacency graph's laplacian be to be considered as
        zero values (number of zero values determines number of clusters, so higher tolerance -> more clusters).
        :param max_island_size: maximum size of the cluster
        """
        self._adjacency_matrix_creator = similarity_attributes_creator
        self.tolerance: float = tolerance
        self.max_island_size: int = max_island_size

    def get_clusters(self, population: List[Phenotype], min_size: int) -> List[List[Phenotype]]:
        """
        Divide the population into clusters based on a similarities between specimen.
        :param population: list of Phenotype objects
        :return: List of Lists of Phenotype objects
        """
        adjacency_matrix = self._adjacency_matrix_creator.get_adjacency_matrix(population)

        # create the graph's laplacian
        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
        graph_laplacian = degree_matrix - adjacency_matrix

        # find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(graph_laplacian)

        # remove imaginary part of complex numbers (imaginary part exists only because of
        # iterative algorithm precision issues)
        eigenvalues, eigenvectors = np.abs(eigenvalues), np.abs(eigenvectors)

        # sort
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
        eigenvalues = eigenvalues[np.argsort(eigenvalues)]

        # Number of clusters is defined by the number of zero eigenvalues of a kneighbors graph laplacian.
        # "If the graph (W) has K connected components, then L has K eigenvectors with an eigenvalue of 0."
        n_clusters = np.sum(np.isclose(eigenvalues, 0, atol=self.tolerance))

        if n_clusters < 2:
            return split_clusters(self.max_island_size, [population])

        U = np.array(eigenvectors[:, 1:n_clusters])
        km = KMeans(init="k-means++", n_clusters=n_clusters)
        km.fit(U)

        clusters: List[List[Phenotype]] = [[] for _ in range(n_clusters)]

        for idx, label in enumerate(km.labels_):
            clusters[label].append(population[idx])

        clusters = split_clusters(self.max_island_size, clusters)

        return [c for c in clusters if len(c) >= min_size]  # filter out too small clusters
