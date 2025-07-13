import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyproj
import rtree
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.ops import transform

log = logging.getLogger("core")


class GetHistoricalDirections:
    """Class responsible for calculating route probabilities."""

    # distance buffer parameters
    ub = 270  # buffer value
    ub0 = 265  # buffer value around centroid with weight equal 1
    assert ub > ub0

    # time buffer parameters
    minute_range = 15  # minutes range to categorize minutes data
    assert 60 % minute_range == 0
    nb_range = (60 // minute_range) * 24  # bins number in one day

    # rolling window parameters
    window_size = 25  # window size must be odd number
    assert window_size % 2 == 1
    window_center = (window_size - 1) // 2
    sigma = 0.4  # for gaussian distribution function

    def __init__(
        self,
        rents_df: pd.DataFrame,
        cell_geometries: List[Polygon],
        source_crs: str = "epsg:4326",
        target_crs: str = "epsg:2180",
    ):
        self._number_cells = len(cell_geometries)
        # Create transformation object. It'll be used to transform cell geometries before creating buffers around them.
        self._transformation = pyproj.Transformer.from_proj(source_crs, target_crs)
        self._rents_df: pd.DataFrame = rents_df
        log.info("preparing date related fields")
        self._rents_df.start_date = pd.to_datetime(rents_df.start_date)
        self._rents_df["weekday"] = rents_df.start_date.apply(lambda d: d.weekday())
        self._rents_df["hour"] = rents_df.start_date.apply(lambda d: d.hour)
        self._rents_df["minute"] = rents_df.start_date.apply(lambda d: d.minute // self.minute_range)
        log.info("sorting by date")
        self._rents_df = self._rents_df.sort_values(by=["start_date"])
        self._cell_geometries: List[Polygon] = cell_geometries
        self._buffers: np.array = None
        self._buffer_sindex = rtree.index.Index(interleaved=True)
        self._distribution: np.array = None
        self._landuse = np.ones(self._number_cells)
        self._directions: List[np.array] = None

    def get_directions_for_week(self) -> np.array:
        """Method returns directions array for week. Axes: weekday-hour-source-destination"""
        log.info("trying to get directions for week")
        if self._directions is None:
            log.info("directions don't exist, preparing now")
            self._prepare_directions()
        log.info("directions ready")
        return self._directions

    def get_directions_by_weekday(self, weekday: int) -> np.array:
        """Method returns directions array for given weekday. Axes: hour-source-destination"""
        log.info(f"trying to get directions for weekday {weekday}")
        if self._directions is None:
            log.info("directions don't exist, preparing now")
            self._prepare_directions()
        log.info("directions ready")
        return self._directions[weekday]

    def _prepare_directions(self) -> None:
        """Method calculates direction arrays for every weekday"""

        # Transform cell geometries
        log.info("transforming cells to local CRS")
        for i in range(self._number_cells):
            self._cell_geometries[i] = transform(self._transformation.transform, self._cell_geometries[i])

        log.info("creating buffer geometries around cells")
        self._buffers = np.array([g.buffer(self.ub) for g in self._cell_geometries])

        log.info("creating spatial index on buffers")

        for i, buffer in enumerate(self._buffers):
            self._buffer_sindex.insert(i, buffer.bounds)

        log.info("creating point geometries of start and end positions")
        start_positions: np.array = self._rents_df.apply(lambda row: Point(row.start_lon, row.start_lat), axis=1).values
        end_positions: np.array = self._rents_df.apply(lambda row: Point(row.end_lon, row.end_lat), axis=1).values
        date_data: np.array = self._rents_df[["weekday", "hour", "minute"]].values

        self._distribution = self._gaussian_distribution(np.linspace(-1, 1, self.window_size), self.sigma)
        self._directions = [
            np.zeros(shape=[self.nb_range, self._number_cells, self._number_cells], dtype=np.float32) for _ in range(7)
        ]

        log.info("transforming start and end positions to local CRS")
        for i, _ in enumerate(start_positions):
            start_positions[i] = transform(self._transformation.transform, start_positions[i])
            end_positions[i] = transform(self._transformation.transform, end_positions[i])

        log.info("calculating weights")
        start_weights = [self._is_contained_and_weighted_distance(p) for p in start_positions]
        end_weights = [self._is_contained_and_weighted_distance(p) for p in end_positions]

        log.info("converting data from processed data to array")
        for start, stop, (weekday, hour, minute) in zip(start_weights, end_weights, date_data):
            for start_cell, start_weight in zip(*start):
                for stop_cell, stop_weight in zip(*stop):
                    timebin_index = hour * self.nb_range // 24 + minute
                    self._directions[weekday][timebin_index, start_cell, stop_cell] += np.float32(
                        (start_weight + stop_weight) / 2
                    )

        for i in range(7):
            log.info(f"calculating trip probabilities for weekday {i}")
            self._directions[i] = self._rolling_window_function(self._directions[i])
            self._directions[i] = self._distribute_probabilities_to_cells_with_zero_probability(self._directions[i])
        self._directions = np.array(self._directions)

    def _is_contained_and_weighted_distance(self, point_geometry: Point) -> Tuple[List[int], List[float]]:
        """Method calculates weighted distances from point (point_geometry)"""
        candidate_buffer_idxes = list(self._buffer_sindex.intersection(point_geometry.bounds))
        buffer_idxes = [idx for idx in candidate_buffer_idxes if self._buffers[idx].intersects(point_geometry)]

        # calculate distance from point to the centroids
        dist_to_centroids = [point_geometry.distance(polygon.centroid) for polygon in self._buffers[buffer_idxes]]
        # calculate weighted distance
        weighted_distance = [1 if dist <= self.ub0 else (self.ub0 / dist) ** 2 for dist in dist_to_centroids]

        return buffer_idxes, weighted_distance

    def _rolling_window_function(self, weighted_array: np.array) -> np.array:
        """Method calculates trip probability distribution."""

        # create base array
        output_array = np.zeros(shape=[self.nb_range, self._number_cells, self._number_cells], dtype=np.float32)

        for idx in range(self.nb_range):
            # The 'left' exception
            if idx < self.window_center:
                zip_ = zip(
                    weighted_array[0 : idx + self.window_center + 1], self._distribution[self.window_center - idx :]
                )
                # The 'right' exception
            elif self.nb_range - idx - 1 < self.window_center:
                zip_ = zip(
                    weighted_array[idx - self.window_center : self.nb_range],
                    self._distribution[self.window_center - idx :],
                )
            # Without any exception
            else:
                zip_ = zip(weighted_array[idx - self.window_center : idx + self.window_center + 1], self._distribution)

            # use created zip_
            output_array[idx] = np.sum([arr * dist for arr, dist in zip_], axis=0)
        hourly_distribution = np.array(
            [
                np.sum(output_array[i : i + self.nb_range // 24], axis=0)
                for i in range(0, self.nb_range, self.nb_range // 24)
            ]
        )

        return hourly_distribution

    def _distribute_probabilities_to_cells_with_zero_probability(self, array: np.array) -> np.array:
        """Method distributes one percent of all trips onto cells with zero trip probability."""

        output_array = np.zeros_like(array)

        for idx, hour in enumerate(array):
            for idy, row in enumerate(hour):
                zero_cells = np.where(row == 0)

                if len(zero_cells[0]) == 0:
                    continue
                else:
                    dist_equation = np.sum(row) * 0.01 / len(zero_cells[0])

                if dist_equation == 0:
                    continue
                else:
                    row[zero_cells] = (row[zero_cells] + dist_equation) * self._landuse[zero_cells]
                    sum_of_row = np.sum(row)
                    output_array[idx, idy, :] = row / sum_of_row

        return output_array

    @staticmethod
    def _gaussian_distribution(x: np.array, sigma: float, mu: float = 0) -> np.array:
        """Method returns probability distribution for x data and sigma value"""

        return (
            1
            / np.sqrt(2 * np.pi * np.power(sigma, 2.0) + 0.00001)
            * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))
        )
