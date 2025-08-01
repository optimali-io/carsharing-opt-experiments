import logging
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from fleet_manager.data_model.config_models import ScienceConfig
from fleet_manager.data_model.zone_model import ActionType

log = logging.getLogger("fleet_manager")


class OptimisationResult(BaseModel):
    best_phenotype: "Phenotype"
    optimization_time: float
    science_config: ScienceConfig

    class Config:
        arbitrary_types_allowed = True


class Action:
    """
    Model of a single service team action.
    """

    def __init__(
        self,
        vehicle_id: int,
        vehicle_foreign_id: str,
        service_team_id: int,
        service_order: int,
        source_details: dict,
        destination_details: dict,
        station_details: dict = None,
    ):
        self.vehicle_id: int = vehicle_id
        self.vehicle_foreign_id: str = vehicle_foreign_id
        self.service_team_id: int = service_team_id
        self.service_order: int = service_order
        self.source_details: dict = source_details
        self.station_details: dict = station_details
        self.destination_details: dict = destination_details

    def __str__(self):
        return (
            "Vehicle_id: {0}\nVehicle_foreign_id: {1}\nService team: {2}\nService order: {3}\n"
            "Source cell: {4}\nStation: {5}\nDestination cell: {6}\n".format(
                self.vehicle_id,
                self.vehicle_foreign_id,
                self.service_team_id,
                self.service_order,
                self.source_details,
                self.station_details,
                self.destination_details,
            )
        )


class Gene:
    """
    Model of a genetic representation of a single service team action.
    """

    def __init__(
        self,
        vehicle_id: int,
        action_type: int,
        source_cell_id: int,
        destination_cell_id: int,
        gas_station_id: int = -1,
        power_station: int = -1,
    ):
        self.vehicle_id: int = vehicle_id
        self.action_type: int = action_type
        self.destination_cell_id: int = destination_cell_id
        self.source_cell_id: int = source_cell_id
        self.position: Tuple[int, int] = (-1, -1)
        self.in_conflict: bool = False

        self.gas_station_id: int = gas_station_id
        self.power_station_id: int = power_station
        self.unplugged_position: Tuple[int, int] = (
            -1,
            -1,
        )  # Uplugged action position for Charging action

    def deepcopy(self):
        """Create deep copy of a Gene object."""
        return Gene(
            self.vehicle_id,
            self.action_type,
            self.source_cell_id,
            self.destination_cell_id,
            self.gas_station_id,
            self.power_station_id,
        )

    def __str__(self):
        if self.power_station_id != -1:
            return "\t\tV: {0}, A: {1}, S: {2}, D: {3}, CH: {4}\n".format(
                self.vehicle_id,
                self.action_type,
                self.source_cell_id,
                self.destination_cell_id,
                self.power_station_id,
            )
        else:
            return "\t\tV: {0}, A: {1}, S: {2}, D: {3}\n".format(
                self.vehicle_id,
                self.action_type,
                self.source_cell_id,
                self.destination_cell_id,
            )


class Chromosome:
    """Genetic representation of a full day work of a single service team."""

    def __init__(
        self,
        kind,
        service_team_index,
        change_gene_callback: Callable[[Gene, Optional[Tuple[int, int]]], None],
        refuel_time: int = 1080,
        relocation_time: int = 480,
    ) -> None:
        self._genes: List[Gene] = []
        self.kind: int = kind  # 1 or 2 person service team
        self.service_team_index: int = service_team_index  # service team id
        self._change_gene_callback: Callable[[Gene], None] = change_gene_callback

        self.total_time: int = 0
        self.approach_distance: int = 0
        self.relocation_distance: int = 0
        self.refuel_time: int = refuel_time
        self.relocation_time: int = relocation_time
        self.relocations_number: int = 0
        self.refuelings_number: int = 0
        self.plugging_number = 0
        self.unplugging_number = 0

    def __getitem__(self, gene_idx: int) -> Gene:
        if gene_idx >= len(self._genes):
            raise IndexError("Index out of range!")
        return self._genes[gene_idx]

    def __setitem__(self, gene_idx: int, gene: Gene) -> None:
        self._genes[gene_idx].position = None
        self._change_gene_callback(self._genes[gene_idx])

        gene.position = (self.service_team_index, gene_idx)
        self._genes[gene_idx] = gene
        self._change_gene_callback(gene)

    def get_length(self) -> int:
        """Return number of actions (genes) for this service team (chromosome)"""
        return len(self._genes)

    def push_gene(self, gene: Gene, index: int = -1) -> None:
        """
        Insert provided gene (action) into genome (list of actions).
        :param gene: Gene object to be inserted
        :param index: locus of a gene
        """
        if index == -1:
            index = len(self._genes)
        self._genes.insert(index, gene)
        for i in range(index, len(self._genes)):
            self._genes[i].position = (self.service_team_index, i)
        self._change_gene_callback(gene)

    def pop_gene(self, index: int) -> Gene:
        """
        Pop Gene (action) from provided index.
        :param index: locus of a gene
        :return: Gene object
        """
        gene = self._genes.pop(index)
        gene.position = None
        self._change_gene_callback(gene)
        if index < len(self._genes):
            for i in range(index, len(self._genes)):
                self._genes[i].position = (self.service_team_index, i)

        return gene

    def copy_from(
        self, source: "Chromosome", from_gene: int = 0, to_gene: int = -1
    ) -> None:
        """
        Copy genes (actions) from source chromosome to self.
        :param source: Chromosome object to copy genes from
        :param from_gene: start locus for copying
        :param to_gene: end locus for copying
        """
        if to_gene == -1:
            to_gene = source.get_length()
        for i in range(from_gene, to_gene):
            gene: Gene = source._genes[i].deepcopy()
            self._genes.append(gene)
            gene.position = (self.service_team_index, len(self._genes) - 1)
            self._change_gene_callback(gene)

    def __str__(self):
        description: str = "\tService team: {0}\n".format(self.service_team_index)
        description += "\tTime: {0}\n".format(self.total_time)
        description += "\tApproach distance: {0}\n".format(self.approach_distance)
        description += "\tRelocation distance: {0}\n".format(self.relocation_distance)
        for gene in self._genes:
            description += gene.__str__()
        return description


class Genome:
    """
    Genetic representation of the actions of all service teams from zone
    """

    def __init__(self, fleet_length: int, parent: Optional["Genome"] = None) -> None:
        self.gene_counter: int = 0  # total number of genes
        self._chromosomes: List[Chromosome] = []

        self.used_genes: List[Optional[Gene]] = [
            None
        ] * fleet_length  # list of Genes for each action (vehicle)
        self.conflicted_genes_1: List[
            Gene
        ] = []  # used during crossover (for 1st type of actions)
        self.conflicted_genes_2: List[
            Gene
        ] = []  # used during crossover (for 2nd type of actions)
        self.backup_genes_1: List[
            Gene
        ] = []  # used during resolving conflicts (for 1st type of actions)
        self.backup_genes_2: List[
            Gene
        ] = []  # used during resolving conflicts (for 2nd type of actions)

        self._crossover_phase: int = 0  # 0 - begin or end part, 1 - middle part

        if parent is None:
            self.number_chromosome_1: int = 0
            self.number_chromosome_2: int = 0
        else:
            self.number_chromosome_1: int = parent.number_chromosome_1
            self.number_chromosome_2: int = parent.number_chromosome_2

            for ch_i in range(parent.number_chromosome_1):
                self._chromosomes.append(
                    Chromosome(
                        1, parent[ch_i].service_team_index, self.change_gene_callback
                    )
                )
            for ch_i in range(
                parent.number_chromosome_1,
                parent.number_chromosome_1 + parent.number_chromosome_2,
            ):
                self._chromosomes.append(
                    Chromosome(
                        2, parent[ch_i].service_team_index, self.change_gene_callback
                    )
                )

    def __getitem__(self, chromosome_idx: int) -> Chromosome:
        return self._chromosomes[chromosome_idx]

    def push_chromosome(self, chromosome: Chromosome) -> None:
        """
        Insert provided Chromosome object to list of chromosomes.
        :param chromosome: Chromosome object to be inserted
        """
        if chromosome.kind == 1:
            self._chromosomes.insert(self.number_chromosome_1, chromosome)
        else:
            self._chromosomes.append(chromosome)

    def change_gene_callback(self, gene: Gene) -> None:
        """
        Callback used every time any Gene object in Genome is changed (added, removed, updated etc.)
        :param gene: Gene object
        """
        if self._crossover_phase == 0:
            # mutation
            if gene.position is None:
                if self.used_genes[gene.vehicle_id] is None:
                    raise AssertionError("Deleting unassigned gene")
                self.used_genes[gene.vehicle_id] = None
                self.gene_counter -= 1
            else:
                if self.used_genes[gene.vehicle_id] is not None:
                    raise AssertionError("Adding assigned gene")
                self.used_genes[gene.vehicle_id] = gene
                self.gene_counter += 1

        elif self._crossover_phase == 1:
            # begin and end part of crossover phase
            if self.used_genes[gene.vehicle_id] is not None:
                for g in self.conflicted_genes_1:
                    if (
                        g.position[0] == gene.position[0]
                        and g.position[1] == gene.position[1]
                    ):
                        raise ValueError("Collided action conflict!")
                if self[gene.position[0]].kind == 1:
                    self.conflicted_genes_1.append(gene)
                else:
                    self.conflicted_genes_2.append(gene)
                gene.in_conflict = True
            else:
                self.used_genes[gene.vehicle_id] = gene
            self.gene_counter += 1
        elif self._crossover_phase == 2:
            # middle part of crossover
            if self.used_genes[gene.vehicle_id] is not None:
                for g in self.conflicted_genes_1:
                    if (
                        g.position[0] == self.used_genes[gene.vehicle_id].position[0]
                        and g.position[1]
                        == self.used_genes[gene.vehicle_id].position[1]
                    ):
                        raise ValueError("Collided action conflict!")
                conflicted_gene: Gene = self.used_genes[gene.vehicle_id]
                conflicted_gene.in_conflict = True
                if self[conflicted_gene.position[0]].kind == 1:
                    self.conflicted_genes_1.append(conflicted_gene)
                else:
                    self.conflicted_genes_2.append(conflicted_gene)

                # self.conflicted_genes_1.append(self.used_genes[gene.vehicle_id])
                # self.used_genes[gene.vehicle_id].in_conflict = True
            self.used_genes[gene.vehicle_id] = gene
            self.gene_counter += 1
        elif self._crossover_phase == 3:
            # resolving conflicts
            if gene.position is None:
                if self.used_genes[gene.vehicle_id] is None:
                    raise AssertionError("Deleting unassigned action")
                self.gene_counter -= 1
            else:
                if self.used_genes[gene.vehicle_id] is None:
                    self.used_genes[gene.vehicle_id] = gene
                    self.gene_counter += 1  # add new gene

    def set_crossover_phase(self, phase: int):
        """
        Sets faze of a crossover where:
        0 - mutation
        1 - begin and end part of a crossover phase
        2 - middle part of a crossover
        3 - resolving conflicts
        :param phase: phase of a crossover
        """
        self._crossover_phase = phase

    def get_chromosome_and_gene_position(self, gene_number: int) -> Tuple[int, int]:
        """
        Get position of a gene as a tuple (chromosome locus, gene locus).
        :param gene_number: number of a gene
        """
        ch_n: int = 0
        g_c: int = 0

        while g_c + self._chromosomes[ch_n].get_length() <= gene_number:
            g_c += self._chromosomes[ch_n].get_length()
            ch_n += 1

        return ch_n, gene_number - g_c

    def check_integrity(self):
        """
        Checks if there are no invalid actions within genes.
        """
        ch_c: int = 0
        g_c: int = 0
        for chromosome in self._chromosomes:
            ch_c = chromosome.get_length()
            g_c += ch_c
            for gene in chromosome._genes:
                if chromosome.kind == 1 and gene.action_type == ActionType.R:
                    log.info("Invalid gene action!")
                if gene.action_type == ActionType.P and gene.power_station_id == -1:
                    log.info("Invalid gene charge station!")

        if g_c != self.gene_counter:
            log.info("Invalid gene counter!")

    def __str__(self):
        description: str = ""
        for chromosome in self._chromosomes:
            description += chromosome.__str__()
        return description


class Phenotype:
    """
    Result of an optimization containing lists of actions for teams and summaries (revenue, costs etc.)
    """

    def __init__(self, genome):
        self.genome: Genome = genome
        self.fitness: float = -1000000
        self.base_revenue: float = 0
        self.diff_revenue: float = 0
        self.cost: float = 0
        self.penalty: float = 0
        self.bonus: float = 0

    def __str__(self):
        return (
            "Genome:\n{0}Base Revenue: {1}\nDiff Revenue: {2}\nCosts: {3}\n"
            "Penalty: {4}\nBonus: {5}\nFitness: {6}\n".format(
                self.genome,
                self.base_revenue,
                self.diff_revenue,
                self.cost,
                self.penalty,
                self.bonus,
                self.fitness,
            )
        )

    def modify_dst_cell_for_chromosome_1(self, zone_data) -> None:
        """
        In current model last destination cell for 1p team must be this team's "depot".
        :param zone_data: ZoneData object
        """
        for ch_i in range(
            self.genome.number_chromosome_1 + self.genome.number_chromosome_2
        ):
            if self.genome[ch_i].kind == 1:
                for g_i in range(self.genome[ch_i].get_length()):
                    if g_i < (self.genome[ch_i].get_length() - 1):
                        self.genome[ch_i][g_i].destination_cell_id = int(
                            self.genome[ch_i][g_i + 1].source_cell_id
                        )
                    else:
                        self.genome[ch_i][g_i].destination_cell_id = int(
                            zone_data.service_team[
                                self.genome[ch_i].service_team_index
                            ].end_cell_id
                        )

    def to_dict(self) -> Dict:
        """
        Creates dict representation of self.
        """
        genome_dict: dict = {}
        for ch_i in range(
            self.genome.number_chromosome_1 + self.genome.number_chromosome_2
        ):
            chromosome_dict: dict = {
                "service_team_id": int(self.genome[ch_i].service_team_index),
                "kind": int(self.genome[ch_i].kind),
                "team_time": int(self.genome[ch_i].total_time),
                "approach_distance": int(self.genome[ch_i].approach_distance),
                "relocation_distance": int(self.genome[ch_i].relocation_distance),
                "genes": [],
            }
            for g_i in range(self.genome[ch_i].get_length()):
                gene: Gene = self.genome[ch_i][g_i]
                gene_dict = {
                    "action": int(gene.action_type),
                    "vehicle": int(gene.vehicle_id),
                    "source_cell_id": int(gene.source_cell_id),
                    "destination_cell_id": int(gene.destination_cell_id),
                }
                chromosome_dict["genes"].append(gene_dict)
            genome_dict[ch_i] = chromosome_dict

        phenotype_dict: dict = {
            "fitness": float(self.fitness),
            "base_revenue": float(self.base_revenue),
            "diff_revenue": float(self.diff_revenue),
            "cost": float(self.cost),
            "penalty": float(self.penalty),
            "bonus": float(self.bonus),
            "genome": genome_dict,
        }

        return phenotype_dict
