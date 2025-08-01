import multiprocessing
from multiprocessing import Process

from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.genetic_model import Phenotype, Genome
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_algorithm.connection_topography import IConnectionTopography, ConnectionTopographyFactory
from fleet_manager.genetic_algorithm.genetic_incubator import Incubator
from fleet_manager.genetic_algorithm.genetic_worker import GeneticWorkerFactory, GeneticWorker
from fleet_manager.genetic_algorithm.runners import IGeneticWorkersRunner
from fleet_manager.genetic_algorithm.spectral_clustering import RevenueCostChromosomeLenSimilarity, SpectralClustering


class GeneticControllerParallelCpu:
    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData):
        self._configuration: GeneticConfiguration = configuration
        self._incubator: Incubator = Incubator(configuration, zone_data)
        self.best_phenotype: Phenotype = Phenotype(Genome)  # empty phenotype
        self.zone_data: ZoneData = zone_data
        self._worker_runner = None
        self._topography: IConnectionTopography = ConnectionTopographyFactory().get_topography(
            configuration.topography_type)
        self.best_generation = 0

    def execute(self) -> None:
        n_workers = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        processes = []
        worker_factory = GeneticWorkerFactory(
            configuration=self._configuration,
            zone_data=self.zone_data,
            return_dict=return_dict,
            n_workers=n_workers
        )
        workers: list[GeneticWorker] = worker_factory.create_workers()
        self._topography.connect_workers(workers)

        for worker in workers:
            p = Process(target=lambda w: w.execute(), args=(worker,))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


        for d in return_dict.values():
            phenotype = d['phenotype']
            if phenotype.fitness > self.best_phenotype.fitness:
                self.best_phenotype = phenotype
                self.best_generation = d['best_generation']

        print(self.best_phenotype)


class GeneticControllerParallelDynamicIslands:
    """Class responsible for GeneticWorker objects generation, execution and saving optimisation results."""

    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData):
        self._configuration: GeneticConfiguration = configuration
        self._incubator: Incubator = Incubator(configuration, zone_data)
        self.best_phenotype: Phenotype = Phenotype(Genome)  # empty phenotype
        similarity = RevenueCostChromosomeLenSimilarity(n_neighbors=configuration.n_neighbors)
        self._spectral_clustering = SpectralClustering(
            similarity_attributes_creator=similarity,
            tolerance=configuration.tolerance,
            max_island_size=configuration.island_population,
        )
        self._epochs = [
            configuration.migration_interval
            for _ in range(int(configuration.maximum_generation_number / configuration.migration_interval))
        ]
        self.zone_data: ZoneData = zone_data
        self._worker_runner = None

    def set_workers_runner(self, worker_runner: IGeneticWorkersRunner):
        """
        Set workers runner
        :param worker_runner: Instance inheriting IGeneticWorkersRunner interface.
        """
        self._worker_runner = worker_runner

    def execute(self) -> None:
        """Run optimisation"""
        if not self._worker_runner:
            raise AttributeError("Worker runner not set!")
        counter: int = 0

        population = self._incubator.population_initialization()

        clusters = self._spectral_clustering.get_clusters(population, min_size=self._configuration.tournament_samples*2)

        for epoch in self._epochs:
            print(f"Epoch {counter}, clusters: {len(clusters)}")
            population = self._worker_runner.run_workers(clusters)
            for phenotype in population:
                if phenotype.fitness > self.best_phenotype.fitness:
                    self.best_phenotype = phenotype
            clusters = self._spectral_clustering.get_clusters(population, min_size=self._configuration.tournament_samples*2)
            counter += epoch
        self.best_phenotype.modify_dst_cell_for_chromosome_1(self.zone_data)