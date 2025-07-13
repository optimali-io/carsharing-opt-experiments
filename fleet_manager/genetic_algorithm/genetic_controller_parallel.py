import logging


from fleet_manager.data_model.genetic_model import Genome, Phenotype
from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.zone_model import ZoneData
from fleet_manager.genetic_algorithm.genetic_evaluator import Evaluator
from fleet_manager.genetic_algorithm.genetic_incubator import Incubator
from fleet_manager.genetic_algorithm.spectral_clustering import RevenueCostChromosomeLenSimilarity, SpectralClustering

log = logging.getLogger("fleet_manager")


class GeneticControllerParallelDynamic:
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

    def set_workers_runner(self, worker_runner):
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

        clusters = self._spectral_clustering.get_clusters(population)

        for epoch in self._epochs:
            population = self._worker_runner.run_workers(clusters)
            for phenotype in population:
                if phenotype.fitness > self.best_phenotype.fitness:
                    self.best_phenotype = phenotype
            clusters = self._spectral_clustering.get_clusters(population)
            counter += epoch
        self.best_phenotype.modify_dst_cell_for_chromosome_1(self.zone_data)
