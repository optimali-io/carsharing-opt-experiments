import itertools
import math
import re
from multiprocessing.connection import Pipe
from typing import List

from fleet_manager.genetic_algorithm.genetic_worker import GeneticWorker, GeneticWorkerIsland, GeneticWorkerCellular


class IConnectionTopography:
    """Interface for all connection topography creators."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def connect_workers(self, workers: List[GeneticWorker]) -> None:
        """Method creates connections between GeneticWorker objects."""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class ConnectionTopographyFactory:
    """Class used to load correct IConnectionTopography object."""

    def __init__(self):
        self._topography_types_dict = {
            'IslandOneWayRing': IslandOneWayRing,
            'IslandTwoWayRing': IslandTwoWayRing,
            'IslandFullyConnected': IslandFullyConnected,
            'IslandStar': IslandStar,
            'IslandLattice': IslandLattice,
            'CellularLinearFixedBorder': CellularLinearFixedBorder,
            'CellularLinearNoBorder': CellularLinearNoBorder,
            'CellularCompactFixedBorder': CellularCompactFixedBorder,
            'CellularCompactNoBorder': CellularCompactNoBorder,
            'CellularDiamondFixedBorder': CellularDiamondFixedBorder,
            'CellularDiamondNoBorder': CellularDiamondNoBorder

        }

    def get_topography(self, topography_type: str) -> IConnectionTopography:
        """Loads an IConnectionTopography object of given type."""

        arg_name = ''
        kwargs = {}

        # Check if numbers in name (Cellular<...> or Star topographies)
        number = re.findall(r'\d+', topography_type)
        if 'Star' in topography_type:
            # 0 is the default value for "center_worker_idx"
            arg_name = 'center_worker_idx'
            kwargs[arg_name] = 0
        elif 'Cellular' in topography_type:
            arg_name = 'neighbor_range'
        if number:
            topography_type = topography_type.split(number[0])[0]
            number = int(number[0])
            kwargs[arg_name] = number

        if topography_type not in self._topography_types_dict.keys():
            raise KeyError(
                f'Topography type "{topography_type}" is incorrect.' +
                f'Possible values: [{", ".join(self._topography_types_dict.keys())}] ')
        return self._topography_types_dict[topography_type](**kwargs)


class IslandOneWayRing(IConnectionTopography):
    """Connects every worker so it can receive from preceding worker and send to next worker."""

    #   w1 - w2
    #  /      \
    # w0       w3
    #  \      /
    #   w5 - w4

    def connect_workers(self, workers: List[GeneticWorkerIsland]) -> None:
        n_workers = len(workers)
        pipes = [Pipe() for _ in range(n_workers)]
        for i, p in enumerate(pipes):
            workers[i - 1].send_connections.append(p[0])
            workers[i].receive_connections.append(p[1])


class IslandTwoWayRing(IConnectionTopography):
    """Connects every worker so it can receive from and send to preceding and next workers."""

    #   w1 - w2
    #  /      \
    # w0       w3
    #  \      /
    #   w5 - w4

    def connect_workers(self, workers: List[GeneticWorkerIsland]) -> None:
        n_workers = len(workers)
        pipes = [Pipe() for _ in range(n_workers)]
        for i, p in enumerate(pipes):
            workers[i - 1].send_connections.append(p[0])
            workers[i - 1].receive_connections.append(p[0])
            workers[i].send_connections.append(p[1])
            workers[i].receive_connections.append(p[1])


class IslandFullyConnected(IConnectionTopography):
    """Connects every worker so it can receive from all other workers."""

    # w0 ---w1
    # | \  /|
    # |  X  |
    # |/  \ |
    # w3---w2

    def connect_workers(self, workers: List[GeneticWorkerIsland]) -> None:
        n_workers = len(workers)
        pairs = itertools.combinations(range(n_workers), 2)
        for worker_id1, worker_id2 in pairs:
            c1, c2 = Pipe()
            workers[worker_id1].send_connections.append(c1)
            workers[worker_id1].receive_connections.append(c1)
            workers[worker_id2].send_connections.append(c2)
            workers[worker_id2].receive_connections.append(c2)


class IslandStar(IConnectionTopography):
    """Connects every worker so it can receive from and send to central worker."""

    # w1    w2
    #  \    /
    #   \  /
    #    w0
    #   /  \
    #  /    \
    # w3    w4

    def __init__(self, center_worker_idx: int):
        self.center_worker_idx = center_worker_idx

    def connect_workers(self, workers: List[GeneticWorkerIsland]) -> None:
        n_workers = len(workers)
        for i in [n for n in range(n_workers) if n != self.center_worker_idx]:
            c1, c2 = Pipe()
            workers[self.center_worker_idx].send_connections.append(c1)
            workers[self.center_worker_idx].receive_connections.append(c1)
            workers[i].send_connections.append(c2)
            workers[i].receive_connections.append(c2)


class IslandLattice(IConnectionTopography):
    """Connects every worker so it can send to and receive from up to four neighbor workers (manhattan metric)."""

    # w0--w1--w2
    # |   |   |
    # w3--w4--w5

    def connect_workers(self, workers: List[GeneticWorkerIsland]) -> None:
        n_workers = len(workers)
        n_rows = int(math.sqrt(n_workers))
        n_columns = int(n_workers / n_rows)
        for row_idx in range(n_rows):
            for col_idx in range(n_columns):
                worker_id = n_columns * row_idx + col_idx
                neighbor_right_row = row_idx
                neighbor_right_col = col_idx + 1

                neighbor_bottom_row = row_idx + 1
                neighbor_bottom_col = col_idx

                if neighbor_right_col < n_columns:
                    neighbor_id = neighbor_right_col + n_columns * neighbor_right_row
                    c1, c2 = Pipe()
                    workers[neighbor_id].send_connections.append(c1)
                    workers[neighbor_id].receive_connections.append(c1)
                    workers[worker_id].send_connections.append(c2)
                    workers[worker_id].receive_connections.append(c2)

                if neighbor_bottom_row < n_rows:
                    neighbor_id = neighbor_bottom_col + n_columns * neighbor_bottom_row
                    c1, c2 = Pipe()
                    workers[neighbor_id].send_connections.append(c1)
                    workers[neighbor_id].receive_connections.append(c1)
                    workers[worker_id].send_connections.append(c2)
                    workers[worker_id].receive_connections.append(c2)


class CellularLinearFixedBorder(IConnectionTopography):
    """
    Connects every specimen "S" up to k=4*r neighbors "N", where "r" is radius. No neighbors over the edges.
    """

    #
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  N  N  S  N  N  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  *  *  N  *  *  *  N
    # *  *  *  *  *  *  *  *  N
    # *  *  *  *  *  *  N  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=True, neighborhood_type='Linear')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellularLinearNoBorder(IConnectionTopography):
    """
    Connects every specimen "S" to k=4*r neighbors "N", where "r" is radius. No edges.
    """

    #
    # *  *  *  *  *  *  *  *  N
    # *  *  *  *  *  *  *  *  N
    # *  *  *  *  N  *  *  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  N  N  S  N  N  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  *  *  N  *  *  *  N
    # *  *  *  *  *  *  *  *  N
    # N  N  *  *  *  *  N  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=False, neighborhood_type='Linear')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellularCompactFixedBorder(IConnectionTopography):
    """
    Connects every specimen "S" to k=(2*r+1)^2-1 neighbors "N", where "r" is radius. No neighbors over the edges.
    """
    #
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  N  S  N  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  N  N
    # *  *  *  *  *  *  *  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=True, neighborhood_type='Compact')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellularCompactNoBorder(IConnectionTopography):
    """
    Connects every specimen "S" to k=(2*r+1)^2-1 neighbors "N", where "r" is radius. No edges.
    """
    #
    # N  *  *  *  *  *  *  N  N
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  N  S  N  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  *  *  *  *  *  *
    # N  *  *  *  *  *  *  N  N
    # N  *  *  *  *  *  *  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=False, neighborhood_type='Compact')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellularDiamondFixedBorder(IConnectionTopography):
    """
    Connects every specimen "S" to every cell within range of r in manhattan metric. No neighbors over the edge.
    """

    # k = 4
    #
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  *  *  *  *  *
    # *  *  *  *  N  *  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  N  N  S  N  N  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  *  N  *  *  *  N
    # *  *  *  *  *  *  *  N  N
    # *  *  *  *  *  *  N  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=True, neighborhood_type='Diamond')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellularDiamondNoBorder(IConnectionTopography):
    """
    Connects every specimen "S" to every cell within range of r in manhattan metric. No neighbors over the edge.
    """

    # k = 4
    #
    # N  *  *  *  *  *  *  N  N
    # *  *  *  *  *  *  *  *  N
    # *  *  *  *  N  *  *  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  N  N  S  N  N  *  *
    # *  *  *  N  N  N  *  *  *
    # *  *  *  *  N  *  *  *  N
    # N  *  *  *  *  *  *  N  N
    # N  N  *  *  *  *  N  N  S

    def __init__(self, neighbor_range: int = None):
        self.neighbor_range = neighbor_range
        self.connector = CellularConnector(neighbor_range=neighbor_range, borders=False, neighborhood_type='Diamond')

    def connect_workers(self, workers: List[GeneticWorkerCellular]) -> None:
        self.connector.connect(workers=workers)


class CellIdxGetter:
    """Class responsible for determining cell idx given row and column position."""

    def __init__(self, borders: bool, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.borders = borders

    def get_cell_idx(self, row_idx: int, col_idx: int) -> int:
        """Get cell index in rectangular grid given row and column index."""
        if self.borders:
            if not (0 <= row_idx < self.n_rows and 0 <= col_idx < self.n_cols):
                return None
        else:
            if row_idx < 0:
                row_idx = self.n_rows + row_idx
            elif row_idx >= self.n_rows:
                row_idx = row_idx - self.n_rows
            if col_idx < 0:
                col_idx = self.n_cols + col_idx
            elif col_idx >= self.n_cols:
                col_idx = col_idx - self.n_cols
        return col_idx + self.n_cols * row_idx


class LinearNeighborhoodGetter:
    """Class responsible for getting list of indices of neighbors for given cell indices."""

    def __init__(self, n_rows: int, n_cols: int, borders: bool, neighbor_range: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.borders = borders
        self.neighbor_range = neighbor_range
        self.index_getter = CellIdxGetter(borders, n_rows, n_cols)

    def get_neighbor_indices(self, row_idx, col_idx) -> List[int]:
        """Returns list of neighbor indices for given row and column."""
        neighbor_ids = []
        for distance in range(1, self.neighbor_range + 1):
            up_row = row_idx - distance
            up_col = col_idx
            bottom_row = row_idx + distance
            bottom_col = col_idx
            left_row = row_idx
            left_col = col_idx - distance
            right_row = row_idx
            right_col = col_idx + distance

            for n_id in [self.index_getter.get_cell_idx(up_row, up_col),
                         self.index_getter.get_cell_idx(bottom_row, bottom_col),
                         self.index_getter.get_cell_idx(right_row, right_col),
                         self.index_getter.get_cell_idx(left_row, left_col)]:
                if n_id:
                    neighbor_ids.append(n_id)

        return neighbor_ids


class CompactNeighborhoodGetter:
    """Class responsible for getting list of indices of neighbors for given cell indices."""

    def __init__(self, n_rows: int, n_cols: int, borders: bool, neighbor_range: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.borders = borders
        self.neighbor_range = neighbor_range
        self.index_getter = CellIdxGetter(borders, n_rows, n_cols)

    def get_neighbor_indices(self, row_idx, col_idx) -> List[int]:
        """Returns list of neighbor indices for given row and column."""
        neighbor_ids = []

        for row in range(row_idx - self.neighbor_range, row_idx + self.neighbor_range + 1):
            for col in range(col_idx - self.neighbor_range, col_idx + self.neighbor_range + 1):
                if row == row_idx and col == col_idx:
                    continue
                n_id = self.index_getter.get_cell_idx(row, col)
                if n_id:
                    neighbor_ids.append(n_id)
        return neighbor_ids


class DiamondNeighborhoodGetter:
    """Class responsible for getting list of indices of neighbors for given cell indices."""

    def __init__(self, n_rows: int, n_cols: int, borders: bool, neighbor_range: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.borders = borders
        self.neighbor_range = neighbor_range
        self.index_getter = CellIdxGetter(borders, n_rows, n_cols)

    def get_neighbor_indices(self, row_idx, col_idx) -> List[int]:
        """Returns list of neighbor indices for given row and column."""
        neighbor_ids = []

        for row in range(row_idx - self.neighbor_range, row_idx + self.neighbor_range + 1):
            for col in range(col_idx - self.neighbor_range, col_idx + self.neighbor_range + 1):
                if (row == row_idx and col == col_idx) or (
                        abs(row_idx - row) + abs(col_idx - col) > self.neighbor_range):  # manhattan distance
                    continue
                n_id = self.index_getter.get_cell_idx(row, col)
                if n_id:
                    neighbor_ids.append(n_id)
        return neighbor_ids


class CellularConnector:
    """Class creates connections between cells."""
    def __init__(self, neighbor_range: int, borders: bool, neighborhood_type: str):
        self.neighbor_range = neighbor_range
        self.borders = borders
        self.neighborhood_getter = CellularConnector._get_neighborhood_getter(neighborhood_type)

    @staticmethod
    def _get_neighborhood_getter(neighborhood_type: str):
        n_getters = {
            'Linear': LinearNeighborhoodGetter,
            'Compact': CompactNeighborhoodGetter,
            'Diamond': DiamondNeighborhoodGetter
        }
        return n_getters[neighborhood_type]

    def connect(self, workers: List[GeneticWorkerCellular]):
        population_sizes = []
        for w in workers:
            population_sizes.append(w.get_population_size())
        full_population_size = sum(population_sizes)
        send_connections = [{} for _ in range(full_population_size)]
        receive_connections = [{} for _ in range(full_population_size)]
        n_rows = int(math.sqrt(full_population_size))
        n_columns = int(full_population_size / n_rows)
        if n_rows * n_columns != full_population_size:
            raise ValueError('Attribute "population_size" for cellular algorithm must be a product of two integers!')
        neighborhood_getter = self.neighborhood_getter(n_rows=n_rows, n_cols=n_columns,
                                                       borders=self.borders, neighbor_range=self.neighbor_range)
        for row_idx in range(n_rows):
            for col_idx in range(n_columns):
                specimen_id = n_columns * row_idx + col_idx
                for n_id in neighborhood_getter.get_neighbor_indices(row_idx, col_idx):
                    if n_id not in receive_connections[specimen_id].keys():
                        c1, c2 = Pipe()
                        receive_connections[specimen_id][n_id] = c1
                        send_connections[specimen_id][n_id] = c1
                        receive_connections[n_id][specimen_id] = c2
                        send_connections[n_id][specimen_id] = c2

        prev_idx = 0
        for worker_id, worker in enumerate(workers):
            worker.receive_connections = receive_connections[prev_idx:prev_idx + population_sizes[worker_id]]
            worker.send_connections = send_connections[prev_idx:prev_idx + population_sizes[worker_id]]
            prev_idx += population_sizes[worker_id]
