from abc import ABC, abstractmethod

from fleet_manager.data_model.genetic_model import Phenotype


class IGeneticWorkersRunner(ABC):
    """
    Class defining interface to implement in derived classed of genetic workers runners.
    """

    @abstractmethod
    def run_workers(self, clusters: list[list[Phenotype]]) -> list[Phenotype]:
        """
        Execute worker runner on each cluster.

        :param clusters: list of lists (subpopulations) of the Phenotype instances.

        :return: flattened list of Phenotypes from all processed clusters
        """

        raise NotImplementedError
