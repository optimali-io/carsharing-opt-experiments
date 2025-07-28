import datetime as dt
import logging
import os
import sys
from copy import deepcopy
from typing import List, Tuple, Any

import numpy as np
import requests
from pydantic import BaseModel

from core.utils.request_helpers import make_get_request

log = logging.getLogger("core")


class HereApiWaypoint(BaseModel):
    """
    Class representing a waypoint in HERE API request.
    """

    lat: float
    lng: float

    def __str__(self):
        return f"{self.lat:.4f},{self.lng:.4f}"


class HereApiMatrixRequest(BaseModel):
    origins: list[HereApiWaypoint]
    destinations: list[HereApiWaypoint]
    regionDefinition: dict[str, str] = {"type": "world"}
    matrixAttributes: list[str] = ["travelTimes", "distances"]
    departureTime: str

class HereMatrix(BaseModel):
    numOrigins: int
    numDestinations: int
    distances: list[int] # meters
    travelTimes: list[int] # seconds
    errors: list[int] | None = None

    def get_distance_time(self, origin_index: int, destination_index: int) -> tuple[int, int]:
        idx = origin_index * self.numDestinations + destination_index
        return self.distances[idx], self.travelTimes[idx]

class HereApiMatrixResponse(BaseModel):
    matrixId: str
    matrix: HereMatrix


class DistanceTimeMatrixCreator:
    """Class responsible for making time and distance matrices."""

    here_url = "https://matrix.router.hereapi.com/v8/matrix"

    def get_distance_and_time_matrices(
        self,
        starts_coor: np.array,
        destinations_coor: np.array,
    ) -> Tuple[np.array, np.array]:
        """
        HERE Matrix Routing API version: 8

        Method returns distance and time matrices of size MxN where M is number of start points and N is
        number of destination points. Because Here API does not allow making requests for matrices
        of size bigger than 15x100, final matrices are created from sub-matrices of that size.

        :param starts_coor: numpy array of size Mx2 containing lon,lat coordinates of starting points
        :param destinations_coor: numpy array of size Nx2 containing lon,lat coordinates of starting points
        :param departure_datetime: datetime.datetime for distance and time calculation
        :return: distances numpy array of size MxN, times numpy array of size MxN
        """
        nb_starts: int = 15 # Hardcoded value, as HERE API allows max 15 origins per request. Non configurable.
        nb_destinations: int = 100 # Hardcoded value, as HERE API allows max 100 destinations per request. Non configurable.
        len_starts = len(starts_coor)
        len_destinations = len(destinations_coor)

        departure_timestring = "any"
        url = self.here_url + "?" + f"apiKey={os.getenv("HERE_API_KEY")}" + "&async=false"

        log.info("prepare main matrices")
        times: np.array = np.zeros(
            shape=[len_starts, len_destinations], dtype=np.uint32
        )
        distances: np.array = np.zeros(
            shape=[len_starts, len_destinations], dtype=np.uint32
        )

        log.info("prepare list of starts idx and destinations idx")
        n_start_arrays = np.ceil(len_starts / nb_starts)
        starts_indices: List[np.array] = np.array_split(
            np.arange(len_starts), n_start_arrays
        )
        n_dest_arrays = np.ceil(len_destinations / nb_destinations)
        destinations_indices: List[np.array] = np.array_split(
            np.arange(len_destinations), n_dest_arrays
        )
        i = 0

        log.info("loading matrices from HERE")
        for s_indices in starts_indices:
            for d_indices in destinations_indices:
                request = HereApiMatrixRequest(
                    origins=[HereApiWaypoint(lat=starts_coor[idx, 1], lng=starts_coor[idx, 0]) for idx in s_indices],
                    destinations=[HereApiWaypoint(lat=destinations_coor[idx, 1], lng=destinations_coor[idx, 0]) for idx in d_indices],
                    departureTime=departure_timestring,
                )
                request_json = request.model_dump()
                response_raw = requests.post(url, json=request_json)

                response = HereApiMatrixResponse.model_validate(response_raw.json())
                matrix: HereMatrix = response.matrix
                # process response
                for i, sidx in enumerate(s_indices):
                    for j, didx in enumerate(d_indices):
                        distance, time = matrix.get_distance_time(origin_index=i, destination_index=j)
                        distances[sidx, didx] = distance
                        times[sidx, didx] = time

                progress = (
                    (i + 1) / (len(starts_indices) * len(destinations_indices)) * 100
                )
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
                        time_cell_station[start_cell, station_id]
                        + time_station_cell[station_id, destination_cell]
                    )
                    if time < min_time:
                        min_time = time
                        min_distance = int(
                            dist_cell_station[start_cell, station_id]
                            + dist_station_cell[station_id, destination_cell]
                        )
                        best_station_id = station_id
                distance_cell_station_cell[start_cell, destination_cell] = min_distance
                time_cell_station_cell[start_cell, destination_cell] = min_time
                id_cell_station_cell[start_cell, destination_cell] = best_station_id
        return distance_cell_station_cell, time_cell_station_cell, id_cell_station_cell
