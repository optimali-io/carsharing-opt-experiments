import random
from typing import List, Tuple

from fleet_manager.data_model.genetic_model import Chromosome, Gene, Genome, Phenotype
from fleet_manager.data_model.zone_model import ActionType, PowerStation, Vehicle, ZoneData


class AugmentedPMX:
    """Class handling augmented partially mapped crossover for genetic algorithm (crossing specimens)."""

    def __init__(self, zone_data: ZoneData):
        self.zone_data = zone_data

    def _sample_cut_point(self, father: Genome, mother: Genome) -> Tuple[int, int]:
        """
        Find two cut points (tuple of indices) that enables splitting first genome into three parts.
        :param father: Genome object
        :param mother: Genome object
        """
        sum_length: int = 0
        chromosome_min_length: List[int] = []
        for i in range(father.number_chromosome_1 + father.number_chromosome_2):
            min_length: int = min(father[i].get_length(), mother[i].get_length())
            if min_length == 0:
                min_length = 1
            sum_length += min_length
            chromosome_min_length.append(min_length)

        cut_point: int = random.randrange(sum_length)
        sum_length = 0
        for ch_n, min_length in enumerate(chromosome_min_length):
            if cut_point < sum_length + min_length:
                return ch_n, cut_point - sum_length
            sum_length += min_length

        raise AssertionError("")

    def copy_gene(
        self, child_chromosome: Chromosome, parent_gene: Gene, second_parent: Genome, second_child: Genome
    ) -> None:
        """
        Copy provided Gene object into the child Chromosome.
        :param child_chromosome: Chromosome object that gene is copied into
        :param parent_gene: copied Gene
        :param second_parent: second parent Chromosome
        :param second_child: second child Chromosome
        """
        child_chromosome.push_gene(parent_gene.deepcopy())

    def augmented_partially_mapped_crossover(self, father: Genome, mother: Genome) -> (Phenotype, Phenotype):
        """
        Run crossover operation on two provided genomes and create two children Phenotypes.
        :param father: Genome object
        :param mother: Genome object
        :return: Tuple of Phenotypes (children)
        """
        cut_point_1: Tuple[int, int] = self._sample_cut_point(father, mother)
        cut_point_2: Tuple[int, int] = self._sample_cut_point(father, mother)

        # swap them if the second is earlier then the first one
        if cut_point_1[0] > cut_point_2[0] or (cut_point_1[0] == cut_point_2[0] and cut_point_1[1] > cut_point_2[1]):
            tmp_point: Tuple[int, int] = cut_point_1
            cut_point_1 = cut_point_2
            cut_point_2 = tmp_point

        # create children genomes
        child_genome_1: Genome = Genome(len(self.zone_data.fleet), father)
        child_genome_2: Genome = Genome(len(self.zone_data.fleet), father)
        child_phenotype_1: Phenotype = Phenotype(child_genome_1)
        child_phenotype_2: Phenotype = Phenotype(child_genome_2)

        # copy the beginning part of genes
        child_genome_1.set_crossover_phase(1)
        child_genome_2.set_crossover_phase(1)

        for ch_i in range(cut_point_1[0]):
            for fg_i in range(father[ch_i].get_length()):
                self.copy_gene(child_genome_1[ch_i], father[ch_i][fg_i], mother, child_genome_2)
            for mg_i in range(mother[ch_i].get_length()):
                self.copy_gene(child_genome_2[ch_i], mother[ch_i][mg_i], father, child_genome_1)

        for pg_i in range(0, cut_point_1[1]):
            self.copy_gene(child_genome_1[cut_point_1[0]], father[cut_point_1[0]][pg_i], mother, child_genome_2)
            self.copy_gene(child_genome_2[cut_point_1[0]], mother[cut_point_1[0]][pg_i], father, child_genome_1)

        # copy the middle part of genes
        child_genome_1.set_crossover_phase(2)
        child_genome_2.set_crossover_phase(2)
        if cut_point_1[0] < cut_point_2[0]:
            for fg_i in range(cut_point_1[1], father[cut_point_1[0]].get_length()):
                child_genome_2[cut_point_1[0]].push_gene(father[cut_point_1[0]][fg_i].deepcopy())
            for mg_i in range(cut_point_1[1], mother[cut_point_1[0]].get_length()):
                child_genome_1[cut_point_1[0]].push_gene(mother[cut_point_1[0]][mg_i].deepcopy())

            for ch_i in range(cut_point_1[0] + 1, cut_point_2[0]):
                for fg_i in range(father[ch_i].get_length()):
                    child_genome_2[ch_i].push_gene(father[ch_i][fg_i].deepcopy())

                for mg_i in range(mother[ch_i].get_length()):
                    child_genome_1[ch_i].push_gene(mother[ch_i][mg_i].deepcopy())

            for pg_i in range(0, cut_point_2[1]):
                child_genome_2[cut_point_2[0]].push_gene(father[cut_point_2[0]][pg_i].deepcopy())
                child_genome_1[cut_point_2[0]].push_gene(mother[cut_point_2[0]][pg_i].deepcopy())
        else:
            for pg_i in range(cut_point_1[1], cut_point_2[1]):
                child_genome_2[cut_point_1[0]].push_gene(father[cut_point_1[0]][pg_i].deepcopy())
                child_genome_1[cut_point_1[0]].push_gene(mother[cut_point_1[0]][pg_i].deepcopy())

        # copy the ending part of genes
        child_genome_1.set_crossover_phase(1)
        child_genome_2.set_crossover_phase(1)

        for fg_i in range(cut_point_2[1], father[cut_point_2[0]].get_length()):
            self.copy_gene(child_genome_1[cut_point_2[0]], father[cut_point_2[0]][fg_i], mother, child_genome_2)
        for mg_i in range(cut_point_2[1], mother[cut_point_2[0]].get_length()):
            self.copy_gene(child_genome_2[cut_point_2[0]], mother[cut_point_2[0]][mg_i], father, child_genome_1)

        for ch_i in range(cut_point_2[0] + 1, father.number_chromosome_1 + father.number_chromosome_2):
            for fg_i in range(father[ch_i].get_length()):
                self.copy_gene(child_genome_1[ch_i], father[ch_i][fg_i], mother, child_genome_2)
            for mg_i in range(mother[ch_i].get_length()):
                self.copy_gene(child_genome_2[ch_i], mother[ch_i][mg_i], father, child_genome_1)

        AugmentedPMX._make_backup_gene_lists(child_genome_1, mother)
        AugmentedPMX._make_backup_gene_lists(child_genome_2, father)

        child_genome_1.check_integrity()
        child_genome_2.check_integrity()

        # resolve collisions
        child_genome_1.set_crossover_phase(3)
        child_genome_2.set_crossover_phase(3)

        self._replace_conflicted_genes_from_backup(child_genome_1)
        self._replace_conflicted_genes_from_backup(child_genome_2)

        child_genome_1.check_integrity()
        child_genome_2.check_integrity()

        child_genome_1.set_crossover_phase(0)
        child_genome_2.set_crossover_phase(0)

        return child_phenotype_1, child_phenotype_2

    @staticmethod
    def _make_backup_gene_lists(child: Genome, parent: Genome) -> None:
        """
        Create a list of backup genes in child Genome for resolving conflicts (if there is a conflict in child genome
        then conflictiong genes will later be replaced by correct genes from backup).
        :param child: child Genome
        :param parent: parent Genome
        """
        for ch_i in range(parent.number_chromosome_1 + parent.number_chromosome_2):
            for g_i in range(parent[ch_i].get_length()):
                gene: Gene = parent[ch_i][g_i]
                if child.used_genes[gene.vehicle_id] is None:
                    if gene.action_type == ActionType.F:
                        child.backup_genes_1.append(gene)
                    child.backup_genes_2.append(gene)

    def _replace_conflicted_genes_from_backup(self, child: Genome) -> None:
        """
        Replace conflicted genes from child Genome by genes from its backup list.
        :param child: child Genome
        """
        cg_1: int = 0  # conflicted gene counter
        bg_1: int = 0  # backup gene counter
        cg_2: int = 0  # conflicted gene counter
        bg_2: int = 0  # backup gene counter

        # replace conflicted genes into parent genes which are not present in the child
        while bg_1 < len(child.backup_genes_1) and cg_1 < len(child.conflicted_genes_1):
            position: Tuple[int, int] = child.conflicted_genes_1[cg_1].position
            child[position[0]][position[1]] = child.backup_genes_1[bg_1].deepcopy()
            cg_1 += 1
            bg_1 += 1

        while bg_2 < len(child.backup_genes_2) and cg_2 < len(child.conflicted_genes_2):
            position: Tuple[int, int] = child.conflicted_genes_2[cg_2].position
            if child.used_genes[child.backup_genes_2[bg_2].vehicle_id] is None:
                child[position[0]][position[1]] = child.backup_genes_2[bg_2].deepcopy()
                cg_2 += 1
            bg_2 += 1

        if cg_1 < len(child.conflicted_genes_1) or cg_2 < len(child.conflicted_genes_2):
            # not all conflicts were resolved
            # so replace conflicted genes into remain available genes
            other_actions_1: List[Vehicle] = []
            # other_actions_2: List[Vehicle] = []

            for vehicle in self.zone_data.fleet:
                if child.used_genes[vehicle.vehicle_index] is None:
                    other_actions_1.append(vehicle)
                    # other_actions_2.append(vehicle)
            if len(other_actions_1) > len(child.conflicted_genes_1) - cg_1 + len(child.conflicted_genes_2) - cg_2:
                sampled_vehicles: List[Vehicle] = random.sample(
                    other_actions_1, len(child.conflicted_genes_1) - cg_1 + len(child.conflicted_genes_2) - cg_2
                )
            else:
                # capacity of chromosomes is equal or even grater than number of all available actions
                # so take all remain genes
                sampled_vehicles: List[Vehicle] = other_actions_1

            sv_1: int = 0
            while sv_1 < len(sampled_vehicles) and cg_1 < len(child.conflicted_genes_1):
                position: Tuple[int, int] = child.conflicted_genes_1[cg_1].position
                destination_cell_id: int = random.sample(self.zone_data.zone_cell, 1)[0]
                new_gene: Gene = Gene(
                    sampled_vehicles[sv_1].vehicle_index,
                    ActionType.F,
                    sampled_vehicles[sv_1].base_cell_id,
                    destination_cell_id,
                )
                child[position[0]][position[1]] = new_gene
                cg_1 += 1
                sv_1 += 1

            while sv_1 < len(sampled_vehicles) and cg_2 < len(child.conflicted_genes_2):
                position: Tuple[int, int] = child.conflicted_genes_2[cg_2].position
                action: int = random.sample(sampled_vehicles[sv_1].allowed_actions, 1)[0]
                if action == ActionType.P:
                    power_station: PowerStation = random.sample(self.zone_data.power_station, 1)[0]
                    new_gene: Gene = Gene(
                        sampled_vehicles[sv_1].vehicle_index,
                        action,
                        sampled_vehicles[sv_1].base_cell_id,
                        power_station.cell_id,
                    )
                    new_gene.power_station_id = power_station.power_station_id
                else:
                    destination_cell_id: int = random.sample(self.zone_data.zone_cell, 1)[0]
                    new_gene: Gene = Gene(
                        sampled_vehicles[sv_1].vehicle_index,
                        action,
                        sampled_vehicles[sv_1].base_cell_id,
                        destination_cell_id,
                    )
                child[position[0]][position[1]] = new_gene
                cg_2 += 1
                sv_1 += 1

            while cg_1 < len(child.conflicted_genes_1):
                # there is no more actions - the rest of collisions have to be deleted
                child_gene: Gene = child.conflicted_genes_1[cg_1]
                child[child_gene.position[0]].pop_gene(child_gene.position[1])
                cg_1 += 1
            while cg_2 < len(child.conflicted_genes_2):
                # there is no more actions - the rest of collisions have to be deleted
                child_gene: Gene = child.conflicted_genes_2[cg_2]
                child[child_gene.position[0]].pop_gene(child_gene.position[1])
                cg_2 += 1
