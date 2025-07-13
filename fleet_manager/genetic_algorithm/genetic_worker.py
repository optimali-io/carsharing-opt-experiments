import pickle
import random
from abc import ABC, abstractmethod
from copy import copy
from itertools import chain
from multiprocessing import Pool
from typing import List, Optional

from retry import retry

from fleet_manager.data_model.genetic_model import Genome, Phenotype
from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_algorithm.augmented_pmx import AugmentedPMX
from fleet_manager.genetic_algorithm.carcinogener import Carcinogener
from fleet_manager.genetic_algorithm.genetic_evaluator import Evaluator


class GeneticWorkerDynamicIsland:
    """
    Class runs the genetic algorithm sequentially for given number of generations between epochs
    """

    def __init__(
        self, configuration: GeneticConfiguration, zone_data: ZoneData, population: List[Phenotype], n_generations: int
    ):
        self.zone_data: ZoneData = zone_data
        self._evaluator: Evaluator = Evaluator(zone_data)
        self._breeder: AugmentedPMX = AugmentedPMX(zone_data)
        self._carcinogener: Carcinogener = Carcinogener(configuration, zone_data)
        self._population: List[Phenotype] = population
        self.best_phenotype: Phenotype = Phenotype(Genome)  # empty phenotype
        self._n_generations: int = n_generations
        self._tournament_samples = configuration.tournament_samples
        self.counter: int = 0

    def _end_condition(self) -> bool:
        """Checks if algorithm should end based on a maximum generation number."""
        self.counter += 1
        return self.counter <= self._n_generations

    def _evaluation(self) -> None:
        """Calculates fitness function values for all phenotypes in population."""
        for phenotype in self._population:
            self._evaluator.evaluate_phenotype(phenotype)
            if self.best_phenotype.fitness < phenotype.fitness:
                self.best_phenotype = phenotype

    def execute(self) -> List[Phenotype]:
        """Execute genetic algorithm and return results."""
        self._evaluation()
        while self._end_condition():
            self._crossover()
            self._mutation()
            self._evaluation()

        return self._population

    def _crossover(self) -> None:
        """Creates new population by crossing elements of a previous population."""
        if len(self._population) < 3:
            return
        new_generation: List[Phenotype] = []
        for i in range(len(self._population) // 2):
            father: Phenotype = self._tournament_selection()
            mother: Phenotype = self._tournament_selection()
            child_1, child_2 = self._breeder.augmented_partially_mapped_crossover(father.genome, mother.genome)
            new_generation.append(child_1)
            new_generation.append(child_2)

        if len(new_generation) < len(self._population):
            new_generation.append(copy(self._population[0]))

        self._population = new_generation

    def _tournament_selection(self) -> Phenotype:
        """Chooses the best phenotype from a subpopulation of size defined in configuration."""
        selected_phenotypes: List[Phenotype] = random.sample(
            self._population, min(self._tournament_samples, len(self._population))
        )
        best_fitness: float = -1000000
        best_phenotype: Optional[Phenotype] = None
        for phenotype in selected_phenotypes:
            if best_fitness < phenotype.fitness:
                best_fitness = phenotype.fitness
                best_phenotype = phenotype

        return best_phenotype

    def _mutation(self, tune_mode: bool = False) -> None:
        """Changes parts of chromosomes with given probability for all population."""
        for phenotype in self._population:
            self._carcinogener.mutate(phenotype.genome, tune_mode)


class IGeneticWorkersRunner(ABC):
    """
    Class defining interface to implement in derived classed of genetic workers runners.
    """

    @abstractmethod
    def run_workers(self, clusters: List[List[Phenotype]]) -> List[Phenotype]:
        """
        Execute worker runner on each cluster.

        :param clusters: list of lists (subpopulations) of the Phenotype instances.

        :return: flattened list of Phenotypes from all processed clusters
        """

        raise NotImplementedError


class CpuGeneticWorkersRunner(IGeneticWorkersRunner):
    """
    Class responsible for creation and control of single epoch data_preparation. It will be used instead of a Huey version
    if Redis will be unavailable or not necessary (unit tests, experiments)
    """

    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData, zone_data_path: str):
        self._configuration: GeneticConfiguration = configuration
        self._zone_data_path: str = zone_data_path
        with open(zone_data_path, "wb") as f:
            pickle.dump(zone_data, f)

    def run_workers(self, clusters: List[List[Phenotype]]) -> List[Phenotype]:
        """
        Execute worker runner on each cluster.

        :param clusters: list of lists (subpopulations) of the Phenotype instances.

        :return: flattened list of Phenotypes from all processed clusters
        """

        with Pool() as p:
            return list(
                chain.from_iterable(
                    p.starmap(run_genetic_worker, [(c, self._configuration, self._zone_data_path) for c in clusters])
                )
            )


@retry(exceptions=(pickle.UnpicklingError), tries=10, delay=0.1)
def run_genetic_worker(
    population: List[Phenotype], configuration: GeneticConfiguration, zone_data_path: str
) -> List[Phenotype]:
    """
    Function creates a GeneticWorkerDynamicIsland instance and executes it. Function exists because
    both huey and cpu runners needs it (multiprocessing.Pool takes a function object as an argument.
    ZoneData is loaded from filesystem because both huey and multiprocessing use pickle to serialize
    function / task arguments, which takes time. Also, serialization is executed sequentially and deserialization
    is executed paralelly.

    :param population: list of a Phenotype instances (previous generation)
    :param configuration: GeneticConfiguration object.
    :param zone_data_path: a path to a serialized ZoneData instance
    :return: list of a Phenotype instances (new generation)
    """
    with open(zone_data_path, "rb") as f:
        zone_data = pickle.load(f)
    genetic_worker = GeneticWorkerDynamicIsland(
        configuration=configuration,
        zone_data=zone_data,
        population=population,
        n_generations=configuration.migration_interval,
    )
    return genetic_worker.execute()
