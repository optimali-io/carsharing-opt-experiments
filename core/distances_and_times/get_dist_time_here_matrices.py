import datetime as dt
import logging
import os
import sys
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from core.utils.request_helpers import make_get_request

log = logging.getLogger("core")


class DistanceTimeMatrixCreator:
    """Class responsible for making time and distance matrices."""

    here_url = "https://matrix.route.ls.hereapi.com/routing/7.2/calculatematrix.json"

    params = {
        "apiKey": os.getenv("HERE_API_KEY"),
        "mode": "balanced;car;traffic:disabled",
        "summaryAttributes": "traveltime,distance,routeId,costfactor",
    }

    def get_distance_and_time_matrices(
        self, starts_coor: np.array = None, destinations_coor: np.array = None, datetime: dt.datetime = None
    ) -> Tuple[np.array, np.array]:
        """
        HERE Matrix Routing API version: 7

        Method returns distance and time matrices of size MxN where M is number of start points and N is
        number of destination points. Because Here API does not allow making requests for matrices
        of size bigger than 15x100, final matrices are created from sub-matrices of that size.

        :param starts_coor: numpy array of size Mx2 containing lon,lat coordinates of starting points
        :param destinations_coor: numpy array of size Nx2 containing lon,lat coordinates of starting points
        :param datetime: datetime.datetime for distance and time calculation
        :return: distances numpy array of size MxN, times numpy array of size MxN
        """
        nb_starts: int = 15
        nb_destinations: int = 100
        len_starts = len(starts_coor)
        len_destinations = len(destinations_coor)

        log.info("prepare main matrices")
        times: np.array = np.zeros(shape=[len_starts, len_destinations], dtype=np.uint32)
        distances: np.array = np.zeros(shape=[len_starts, len_destinations], dtype=np.uint32)

        log.info("prepare list of starts idx and destinations idx")
        n_start_arrays = np.ceil(len_starts / nb_starts)
        starts_indices: List[np.array] = np.array_split(np.arange(len_starts), n_start_arrays)
        n_dest_arrays = np.ceil(len_destinations / nb_destinations)
        destinations_indices: List[np.array] = np.array_split(np.arange(len_destinations), n_dest_arrays)
        i = 0

        log.info("loading matrices from HERE")
        for s_indices in starts_indices:
            for d_indices in destinations_indices:

                query_params = deepcopy(self.params)

                for request_idx, coords_idx in enumerate(s_indices):
                    query_params[
                        f"start{request_idx}"
                    ] = f"{starts_coor[coords_idx, 1]:.4f},{starts_coor[coords_idx, 0]:.4f}"

                for request_idx, coords_idx in enumerate(d_indices):
                    query_params[
                        f"destination{request_idx}"
                    ] = f"{destinations_coor[coords_idx, 1]:.4f},{destinations_coor[coords_idx, 0]:.4f}"
                query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
                response = make_get_request(self.here_url + query_string)
                response = response["response"]["matrixEntry"]

                # process response
                for data in response:
                    start_id = data["startIndex"]
                    destination_id = data["destinationIndex"]
                    times[s_indices[start_id], d_indices[destination_id]] = data["summary"]["travelTime"]
                    distances[s_indices[start_id], d_indices[destination_id]] = data["summary"]["distance"]

                progress = (i + 1) / (len(starts_indices) * len(destinations_indices)) * 100
                sys.stdout.write("\rDownloaded: {:.1f}%, Nb: {}".format(progress, i))
                sys.stdout.flush()
                i += 1
        return distances, times

    @staticmethod
    def get_cell_station_cell_matrices(
        dist_cell_station: np.array,
        time_cell_station: np.array,
        dist_station_cell: np.array,
        time_station_cell: np.array,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Method creates matrices of distances, times and station ids between zone cells with gas station as waypoint.
        Distances, times and station ids are calculated as shortest route from cell A to cell B through a gas station.

        :param dist_cell_station: numpy array of size n_cells x n_gas_stations with distances between
        cells and gas stations
        :param time_cell_station: numpy array of size n_cells x n_gas_stations with times between
        cells and gas stations
        :param dist_station_cell: numpy array of size n_gas_stations x n_cells with distances between
        gas stations and cells
        :param time_station_cell: numpy array of size n_gas_stations x n_cells with times between
        gas stations and cells

        :return 3 numpy arrays of size n_cells x n_cells with distances, times and station ids
        """
        n_cells: int = dist_cell_station.shape[0]
        n_stations: int = dist_cell_station.shape[1]

        log.info("prepare main matrices")
        distance_cell_station_cell: np.array = np.zeros((n_cells, n_cells))
        time_cell_station_cell: np.array = np.zeros((n_cells, n_cells))
        id_cell_station_cell: np.array = np.zeros((n_cells, n_cells))

        log.info("concatenating matrices")
        for start_cell in range(n_cells):
            for destination_cell in range(n_cells):
                min_time = 10000000
                min_distance = 10000000
                best_station_id = -1
                for station_id in range(n_stations):
                    time = int(
                        time_cell_station[start_cell, station_id] + time_station_cell[station_id, destination_cell]
                    )
                    if time < min_time:
                        min_time = time
                        min_distance = int(
                            dist_cell_station[start_cell, station_id] + dist_station_cell[station_id, destination_cell]
                        )
                        best_station_id = station_id
                distance_cell_station_cell[start_cell, destination_cell] = min_distance
                time_cell_station_cell[start_cell, destination_cell] = min_time
                id_cell_station_cell[start_cell, destination_cell] = best_station_id
        return distance_cell_station_cell, time_cell_station_cell, id_cell_station_cell
