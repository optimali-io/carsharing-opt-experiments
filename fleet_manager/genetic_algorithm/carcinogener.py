import random
from typing import List, Tuple

import numpy as np

from fleet_manager.data_model.genetic_model import ActionType, Gene, Genome
from fleet_manager.data_model.config_models import GeneticConfiguration
from fleet_manager.data_model.zone_model import PowerStation, ZoneData


class Carcinogener:
    """Class responsible for running mutation operations in genetic algorithm."""

    def __init__(self, configuration: GeneticConfiguration, zone_data: ZoneData):
        self.configuration: GeneticConfiguration = configuration
        self.zone_data: ZoneData = zone_data

    def mutate(self, genome: Genome, tune_mode: bool = False, base_destination: List[int] = None):
        """
        Mutate provided Genome object.
        :param genome: Genome object
        :param tune_mode: bool, if True then only mutations from tune model list will be chosen
        :param base_destination: List[int] of cell ids that can be chosen as a new destination
        """
        # sample the number of gene mutation
        expected_number_of_mutations: float = genome.gene_counter * self.configuration.probability_gene_mutation
        number_of_mutations: int = np.random.poisson(expected_number_of_mutations)

        for k in range(number_of_mutations):
            genome.check_integrity()
            try:
                gene_number: int = random.randrange(genome.gene_counter)
            except ValueError:
                return
            position: Tuple[int, int] = genome.get_chromosome_and_gene_position(gene_number)

            if tune_mode:
                mutation_type: List[str] = random.choices(
                    self.configuration.tuned_mutation_types, weights=self.configuration.tuned_weight_mutation_types
                )
            else:
                mutation_type: List[str] = random.choices(
                    self.configuration.mutation_types, weights=self.configuration.weight_mutation_types
                )
            if mutation_type[0] == "move":
                if genome[position[0]].get_length() > 1:
                    gene_number: int = random.randrange(genome.gene_counter)
                    move_position: Tuple[int, int] = genome.get_chromosome_and_gene_position(gene_number)

                    if (position[0] != move_position[0] or position[1] != move_position[1]) and (
                        genome[position[0]][position[1]].action_type == ActionType.F
                        or genome[position[0]].kind == genome[move_position[0]].kind
                    ):
                        genome[move_position[0]].push_gene(genome[position[0]].pop_gene(position[1]), move_position[1])

            elif mutation_type[0] == "swap":
                gene_number: int = random.randrange(genome.gene_counter)
                swap_position: Tuple[int, int] = genome.get_chromosome_and_gene_position(gene_number)

                if (position[0] != swap_position[0] or position[1] != swap_position[1]) and (
                    (
                        genome[position[0]][position[1]].action_type == ActionType.F
                        and genome[swap_position[0]][swap_position[1]].action_type == ActionType.F
                    )
                    or genome[position[0]].kind == genome[swap_position[0]].kind
                ):

                    if position[0] < swap_position[0] or (
                        position[0] == swap_position[0] and position[1] < swap_position[1]
                    ):
                        tmp_position: Tuple[int, int] = position
                        position = swap_position
                        swap_position = tmp_position

                    gene_1: Gene = genome[position[0]].pop_gene(position[1])
                    gene_2: Gene = genome[swap_position[0]].pop_gene(swap_position[1])
                    genome[swap_position[0]].push_gene(gene_1, swap_position[1])
                    genome[position[0]].push_gene(gene_2, position[1])

            elif mutation_type[0] == "change_dest":
                genome[position[0]][position[1]].destination_cell_id = random.sample(base_destination, 1)[0]

            elif mutation_type[0] == "add":
                if genome.gene_counter < len(self.zone_data.fleet):
                    if genome[position[0]].kind == 1:
                        # get vehicle to refuel from list
                        vehicle_id = None
                        for vid in self.zone_data.fleet_ids_to_refuel:
                            if genome.used_genes[vid] is None:
                                vehicle_id = vid
                                break
                        if vehicle_id is not None:
                            action: int = ActionType.F
                            source_cell_id = self.zone_data.fleet[vehicle_id].base_cell_id
                            destination_cell_id = random.sample(self.zone_data.allowed_cells_ps[source_cell_id], 1)[0]
                            # destination_cell_id: int = random.sample(self.zone_data.zone_cell, 1)[0]
                            gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                        else:
                            # no vehicles to refuel or all vehicles to refuel already in genome
                            return
                    else:
                        # below loop will be inefficient if capacity of service teams is similar to size of fleet
                        # but it should not be
                        vehicle_id: int = random.randrange(len(self.zone_data.fleet))
                        while genome.used_genes[vehicle_id] is not None:
                            vehicle_id = random.randrange(len(self.zone_data.fleet))
                        source_cell_id = self.zone_data.fleet[vehicle_id].base_cell_id
                        action: int = random.sample(self.zone_data.fleet[vehicle_id].allowed_actions, 1)[0]
                        if action == ActionType.P:
                            power_station: PowerStation = random.sample(self.zone_data.power_station, 1)[0]
                            gene: Gene = Gene(vehicle_id, action, source_cell_id, power_station.cell_id)
                            gene.power_station_id = power_station.power_station_id
                        elif action == ActionType.F:
                            destination_cell_id = random.sample(self.zone_data.allowed_cells_ps[source_cell_id], 1)[0]
                            gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                        else:
                            # destination_cell_id: int = random.sample(self.zone_data.zone_cell, 1)[0]
                            destination_cell_id = random.sample(self.zone_data.allowed_cells[source_cell_id], 1)[0]
                            gene = Gene(vehicle_id, action, source_cell_id, destination_cell_id)
                    genome[position[0]].push_gene(gene, position[1])

            elif mutation_type[0] == "del":
                if genome[position[0]].get_length() >= 1:
                    genome[position[0]].pop_gene(position[1])

            elif mutation_type[0] == "change":
                gene: Gene = genome[position[0]][position[1]]
                if random.random() < self.configuration.probability_action_change:
                    if genome[position[0]].kind == 1:
                        action: int = ActionType.F
                    else:
                        action: int = random.sample(self.zone_data.fleet[gene.vehicle_id].allowed_actions, 1)[0]
                        if action == ActionType.P and gene.power_station_id == -1:
                            power_station: PowerStation = random.sample(self.zone_data.power_station, 1)[0]
                            gene.destination_cell_id = power_station.cell_id
                            gene.power_station_id = power_station.power_station_id
                    gene.action_type = action
                else:
                    # destination change
                    if gene.action_type == ActionType.P:
                        power_station: PowerStation = random.sample(self.zone_data.power_station, 1)[0]
                        gene.power_station_id = power_station.power_station_id
                        gene.destination_cell_id = power_station.cell_id
                    else:
                        # destination_cell_id: int = random.sample(self.zone_data.zone_cell, 1)[0]
                        destination_cell_id = random.sample(self.zone_data.allowed_cells[gene.source_cell_id], 1)[0]
                        gene.destination_cell_id = destination_cell_id
            else:
                raise ValueError("Unknown type of mutation: {0}".format(mutation_type[0]))
