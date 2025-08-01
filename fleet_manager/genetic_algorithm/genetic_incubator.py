import random
from typing import List

from fleet_manager.data_model.genetic_model import ActionType, Chromosome, Gene, Genome, Phenotype
from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.zone_model import PowerStation, ZoneData


class Incubator:
    """
    Incubator randomly initializes first population of individuals for genetic algorithm.

    :param configuration: genetic configuration
    :param zone_data: zone data
    """

    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData) -> None:
        self.configuration: GeneticConfiguration = configuration
        self.zone_data: ZoneData = zone_data

    def population_initialization(self) -> List[Phenotype]:
        """Create initial population composed of random actions."""
        g: GeneticConfiguration = self.configuration
        c: ZoneData = self.zone_data
        population: List[Phenotype] = []

        for i in range(g.population_size):
            vehicle_ids: List[int] = [v.vehicle_index for v in c.fleet]
            vehicle_to_refuel_ids: List[int] = c.fleet_ids_to_refuel
            genome: Genome = Genome(len(c.fleet))
            for service_team in self.zone_data.service_team:
                st_id: int = self.zone_data.service_team_id_by_index_dict[service_team.service_team_id]
                tw: int = service_team.planned_time_work_minutes
                team_number: int = service_team.service_team_kind
                tw *= 1 + random.randint(-g.chromosome_deviation, g.chromosome_deviation) / 100

                if team_number == 1:
                    chromosome = Chromosome(
                        1,
                        st_id,
                        genome.change_gene_callback,
                        refuel_time=service_team.refuel_time_seconds,
                        relocation_time=service_team.relocation_time_seconds,
                    )
                    action_number: int = int(tw / c.action_duration_1)
                    if action_number < 1:
                        action_number = 1
                    for ac in range(action_number):
                        if len(vehicle_ids) > 0 and len(vehicle_to_refuel_ids) > 0:
                            idx = random.randrange(len(vehicle_to_refuel_ids))
                            vehicle_id = vehicle_to_refuel_ids.pop(idx)
                            vehicle_ids.remove(vehicle_id)
                            source_cell_id = c.fleet[vehicle_id].base_cell_id
                            destination_cell_id = random.sample(c.allowed_cells_ps[source_cell_id], 1)[0]
                            gene = Gene(vehicle_id, ActionType.F, source_cell_id, destination_cell_id)
                            chromosome.push_gene(gene)
                    genome.push_chromosome(chromosome)
                    genome.number_chromosome_1 += 1
                elif team_number == 2:
                    chromosome = Chromosome(
                        2,
                        st_id,
                        genome.change_gene_callback,
                        refuel_time=service_team.refuel_time_seconds,
                        relocation_time=service_team.relocation_time_seconds,
                    )
                    action_number = int(tw / c.action_duration_2)
                    if action_number < 1:
                        action_number = 1
                    for ac in range(action_number):
                        if len(vehicle_ids) > 0:
                            idx = random.randrange(len(vehicle_ids))
                            vehicle_id = vehicle_ids.pop(idx)
                            action = random.sample(c.fleet[vehicle_id].allowed_actions, 1)[0]
                            source_cell_id = c.fleet[vehicle_id].base_cell_id
                            if action == ActionType.P:
                                power_station: PowerStation = random.sample(c.power_station, 1)[0]
                                gene: Gene = Gene(vehicle_id, action, source_cell_id, power_station.cell_id)
                                gene.power_station_id = power_station.power_station_id
                            elif action == ActionType.C:
                                destination_cell_id = random.sample(c.allowed_cells_ps[source_cell_id], 1)[0]
                                gene: Gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                                power_station: PowerStation = random.sample(c.power_station, 1)[0]
                                gene.power_station_id = power_station.power_station_id
                            elif action == ActionType.F:
                                destination_cell_id = random.sample(c.allowed_cells_ps[source_cell_id], 1)[0]
                                gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                            else:

                                destination_cell_id = random.sample(c.allowed_cells[source_cell_id], 1)[0]
                                gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                            chromosome.push_gene(gene)
                    genome.push_chromosome(chromosome)
                    genome.number_chromosome_2 += 1
                else:
                    raise ValueError("Unsupported team number")

            phenotype = Phenotype(genome)
            population.append(phenotype)

        return population
