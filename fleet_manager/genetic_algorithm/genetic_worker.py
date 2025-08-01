import pickle
import random
from copy import copy
from multiprocessing.connection import Connection
from typing import Optional
import numpy as np
from retry import retry

from fleet_manager.data_model.genetic_model import Genome, Phenotype
from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_algorithm.augmented_pmx import AugmentedPMX
from fleet_manager.genetic_algorithm.carcinogener import Carcinogener
from fleet_manager.genetic_algorithm.genetic_evaluator import Evaluator
from fleet_manager.genetic_algorithm.genetic_incubator import Incubator


class GeneticWorker:
    """
    Class runs the genetic algorithm sequentially for given number of generations between epochs
    """

    def __init__(
        self,
        configuration: GeneticConfiguration,
        zone_data: ZoneData,
        population: list[Phenotype],
        n_generations: int,
        worker_id: int = 0,
        return_dict: dict = None,
    ):
        self.zone_data: ZoneData = zone_data
        self.configuration: GeneticConfiguration = configuration
        self.evaluator: Evaluator = Evaluator(zone_data)
        self.breeder: AugmentedPMX = AugmentedPMX(zone_data)
        self.carcinogener: Carcinogener = Carcinogener(configuration, zone_data)
        self.population: list[Phenotype] = population
        self.best_phenotype: Phenotype = Phenotype(Genome)  # empty phenotype
        self.n_generations: int = n_generations
        self.counter: int = 0
        self.return_dict: dict = return_dict
        self.worker_id: int = worker_id
        self.best_generation: int = 0  # generation when optimum was found

    def execute(self) -> list[Phenotype]:
        """Execute genetic algorithm and return results."""
        self._evaluation()
        while self._end_condition():
            self._crossover()
            self._mutation()
            self._evaluation()

        return self.population

    def get_population_size(self) -> int:
        """Return size of worker's population."""
        return len(self.population)

    def _end_condition(self) -> bool:
        """Checks if algorithm should end based on a maximum generation number."""
        self.counter += 1
        return self.counter <= self.n_generations

    def _evaluation(self) -> None:
        """Calculates fitness function values for all phenotypes in population."""
        for phenotype in self.population:
            self.evaluator.evaluate_phenotype(phenotype)
            if self.best_phenotype.fitness < phenotype.fitness:
                self.best_phenotype = phenotype

    def _crossover(self) -> None:
        """Creates new population by crossing elements of a previous population."""
        if len(self.population) < 3:
            return
        new_generation: list[Phenotype] = []
        for i in range(len(self.population) // 2):
            father: Phenotype = self._tournament_selection()
            mother: Phenotype = self._tournament_selection()
            child_1, child_2 = self.breeder.augmented_partially_mapped_crossover(
                father.genome, mother.genome
            )
            new_generation.append(child_1)
            new_generation.append(child_2)

        if len(new_generation) < len(self.population):
            new_generation.append(copy(self.population[0]))

        self.population = new_generation

    def _tournament_selection(self) -> Phenotype:
        """Chooses the best phenotype from a subpopulation of size defined in configuration."""
        selected_phenotypes: list[Phenotype] = random.sample(
            self.population, self.configuration.tournament_samples
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
        for phenotype in self.population:
            self.carcinogener.mutate(phenotype.genome, tune_mode)

    def _migrate(self):
        raise NotImplementedError()


class GeneticWorkerDynamicIsland(GeneticWorker):
    def _migrate(self):
        pass


class GeneticWorkerIsland(GeneticWorker):
    """
    Class runs the genetic algorithm sequentially and exchanges parts of population
    with other island workers through its connections.
    """

    def __init__(self, *args, **kwargs):
        super(GeneticWorkerIsland, self).__init__(*args, **kwargs)
        self.send_connections: list[Connection] = []
        self.receive_connections: list[Connection] = []

    def execute(self) -> None:
        """Execute genetic algorithm and store results in multiprocessing.managers.DictProxy object."""
        self._evaluation()
        while self._end_condition():
            self._crossover()
            self._mutation()
            self._evaluation()
            self._migrate()

        self.return_dict[self.worker_id] = {
            "phenotype": self.best_phenotype,
            "best_generation": self.best_generation,
        }
        self._close_connections()

    def _mutation(self, tune_mode: bool = False) -> None:
        """Changes parts of chromosomes with given probability for all population."""
        for phenotype in self.population:
            self.carcinogener.mutate(phenotype.genome, tune_mode)

    def _migrate(self):
        if self.counter % self.configuration.migration_interval == 0:
            print(f"Migrating at generation {self.counter} on worker {self.worker_id}")
            migration_size = int(
                self.configuration.migration_rate * self.configuration.population_size
            )
            sorted_population = sorted(
                self.population, key=lambda phenotype: phenotype.fitness, reverse=True
            )
            received_phenotypes: list[Phenotype] = []
            for i in range(len(self.send_connections)):
                send_conn = self.send_connections[i]
                send_conn.send(sorted_population[:migration_size])
            for i in range(len(self.receive_connections)):
                receive_connection = self.receive_connections[i]
                res = receive_connection.recv()
                received_phenotypes.extend(res)
            phenotypes_to_accept = sorted(
                received_phenotypes, key=lambda p: p.fitness, reverse=True
            )[:migration_size]
            sorted_population[-migration_size:] = phenotypes_to_accept
            self.population = sorted_population

    def _close_connections(self):
        """Closes all opened connections."""
        for c in self.receive_connections:
            c.close()
        for c in self.send_connections:
            c.close()


class GeneticWorkerCellular(GeneticWorker):
    """
    Class runs the genetic algorithm sequentially and crosses every specimen with specimens from given neighborhood.
    """

    def __init__(self, *args, **kwargs):
        super(GeneticWorkerCellular, self).__init__(*args, **kwargs)
        self.send_connections: list[dict[str, Connection]] = []
        self.receive_connections: list[dict[str, Connection]] = []

    def execute(self) -> None:
        """Execute genetic algorithm and store results in multiprocessing.managers.DictProxy object."""
        self._evaluation()
        self._send_phenotypes_to_neighbors()

        while self._end_condition():
            if self.counter % 100 == 0:
                print(
                    f"Generation: {self.counter} best fitness: {self.best_phenotype.fitness} cpu: {self.worker_id}"
                )
            self._genetic_operations()
            self._send_phenotypes_to_neighbors()

        self.return_dict[self.worker_id] = {
            "phenotype": self.best_phenotype,
            "best_generation": self.best_generation,
        }
        self._close_connections()

    def _genetic_operations(self) -> None:
        """Creates new population by crossing elements of a previous population."""
        for i in range(len(self.population)):
            specimen: Phenotype = self.population[i]
            # crossover
            p1, p2 = self._tournament_selection_cellular(i)
            try:
                if specimen.fitness >= p1.fitness and specimen.fitness >= p2.fitness:
                    pass
                elif p1.fitness < specimen.fitness <= p2.fitness:
                    p1 = specimen
                elif p2.fitness < specimen.fitness <= p1.fitness:
                    p2 = specimen
            except Exception as e:
                raise e
            child_1, child_2 = self.breeder.augmented_partially_mapped_crossover(
                p1.genome, p2.genome
            )

            # mutation
            self.carcinogener.mutate(child_1.genome)
            self.carcinogener.mutate(child_2.genome)

            # evaluation
            self.evaluator.evaluate_phenotype(child_1)
            self.evaluator.evaluate_phenotype(child_2)

            if child_1.fitness > child_2.fitness:
                current_best = child_1
            else:
                current_best = child_2
            if current_best.fitness > self.best_phenotype.fitness:
                self.best_phenotype = current_best
            self.population[i] = current_best

    def _tournament_selection_cellular(
        self, specimen_idx: int
    ) -> tuple[Phenotype, Phenotype]:
        """Chooses two best phenotypes from subpopulation from specimen's neighborhood."""
        selected_phenotypes: list[Phenotype] = [
            c.recv() for c in self.receive_connections[specimen_idx].values()
        ]
        if len(selected_phenotypes) < 2:
            print("asd")
        best_fitness: float = -1000000
        second_fitness: float = -1000000
        best_phenotype: Optional[Phenotype] = None
        second_phenotype: Optional[Phenotype] = None
        for phenotype in selected_phenotypes:
            if best_fitness < phenotype.fitness:
                best_fitness, second_fitness = phenotype.fitness, best_fitness
                best_phenotype, second_phenotype = phenotype, best_phenotype
            elif second_fitness < phenotype.fitness:
                second_fitness = phenotype.fitness
                second_phenotype = phenotype

        return best_phenotype, second_phenotype

    def _close_connections(self):
        """Closes all opened connections."""
        for i in range(len(self.receive_connections)):
            for c in self.receive_connections[i].values():
                c.close()
        for i in range(len(self.send_connections)):
            for c in self.send_connections[i].values():
                c.close()

    def _send_phenotypes_to_neighbors(self) -> None:
        for i, p in enumerate(self.population):
            for c in self.send_connections[i].values():
                c.send(p)


class GeneticWorkerFactory:
    """Class responsible for creating GeneticWorker Objects according to given configuration."""

    def __init__(
        self,
        configuration: GeneticConfiguration,
        zone_data: ZoneData,
        return_dict: dict,
        n_workers: int,
    ):
        self.configuration: GeneticConfiguration = configuration
        self.zone_data: ZoneData = zone_data
        self.return_dict: dict = return_dict
        self.n_workers: int = n_workers

    def create_workers(self) -> list[GeneticWorker]:
        workers: list[GeneticWorker] = []
        topography_type = self.configuration.topography_type
        if "Cellular" in topography_type:
            pop_size = int(self.configuration.population_size / self.n_workers)
            remainder = self.configuration.population_size - self.n_workers * pop_size
            for i in range(self.n_workers):
                conf = copy(self.configuration)
                conf.population_size = pop_size
                if i == 0:
                    conf.population_size += remainder

                incubator: Incubator = Incubator(self.configuration, self.zone_data)

                workers.append(
                    GeneticWorkerCellular(
                        zone_data=self.zone_data,
                        configuration=conf,
                        return_dict=self.return_dict,
                        worker_id=i,
                        population=incubator.population_initialization(),
                        n_generations=self.configuration.maximum_generation_number,
                    )
                )
        elif "Island" in topography_type:
            self.configuration.population_size = self.configuration.island_population
            incubator: Incubator = Incubator(self.configuration, self.zone_data)
            for i in range(self.n_workers):
                workers.append(
                    GeneticWorkerIsland(
                        zone_data=self.zone_data,
                        configuration=self.configuration,
                        return_dict=self.return_dict,
                        worker_id=i,
                        population=incubator.population_initialization(),
                        n_generations=self.configuration.maximum_generation_number,
                    )
                )
        return workers


@retry(exceptions=pickle.UnpicklingError, tries=10, delay=0.1)
def run_genetic_worker(
    population: list[Phenotype],
    configuration: GeneticConfiguration,
    zone_data_path: str,
) -> list[Phenotype]:
    """
    Function creates a GeneticWorkerDynamicIsland instance and executes it. Function exists because
    both huey and cpu runners needs it (multiprocessing.Pool takes a function object as an argument).
    ZoneData is loaded from filesystem because both huey and multiprocessing use pickle to serialize
    function / task arguments, which takes time. Also, serialization is executed sequentially and deserialization
    is executed in parallel.

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
