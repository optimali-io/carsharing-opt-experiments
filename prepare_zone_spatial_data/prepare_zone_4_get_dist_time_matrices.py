import datetime as dt
import logging
import os

import fiona
import numpy as np
import pandas as pd

import zone_creator_config as config
from core.distances_and_times.get_dist_time_here_matrices import DistanceTimeMatrixCreator

log = logging.getLogger(__name__)


def create_dist_time_matrices(
    snapping_points_path: str,
    petrol_stations_path: str,
    output_dir: str,
) -> None:
    """
    Function executes creation of all necessary matrices for the zone.
    It creates distance-cell-cell, time-cell-cell, distance-cell-station-cell, time-cell-station-cell and
    id-cell-station-cell matrices and saves them in "output_dir" directory.
    :param snapping_points_path: path to shapefile with snapping points
    :param petrol_stations_path: path to csv file with petrol stations
    :param output_dir: path to directory where all matrices will be saved
    :return: None
    """
    with fiona.open(snapping_points_path, "r") as shp:
        snap_points = sorted(list(shp), key=lambda f: f["properties"]["id"])
    here_request: DistanceTimeMatrixCreator = DistanceTimeMatrixCreator()
    cells_coords: np.array = np.array(
        [f["geometry"]["coordinates"] for f in snap_points]
    )
    petrol_stations_df: pd.DataFrame = pd.read_csv(petrol_stations_path, sep=";")
    petrol_stations_coords: np.array = petrol_stations_df[["Lon", "Lat"]].values

    print("downloading distance and time cell-cell")
    dist_cell_cell, time_cell_cell = here_request.get_distance_and_time_matrices(
        starts_coor=cells_coords, destinations_coor=cells_coords
    )
    print("downloading distance and time cell-station")
    dist_cell_station, time_cell_station = here_request.get_distance_and_time_matrices(
        starts_coor=cells_coords,
        destinations_coor=petrol_stations_coords,
    )
    print("downloading distance and time station-cell")
    dist_station_cell, time_station_cell = here_request.get_distance_and_time_matrices(
        starts_coor=petrol_stations_coords,
        destinations_coor=cells_coords,
    )
    print("creating distances and time cell-station-cell")
    dist_cell_station_cell, time_cell_station_cell, id_cell_station_cell = (
        here_request.get_cell_station_cell_matrices(
            dist_cell_station=dist_cell_station,
            time_cell_station=time_cell_station,
            dist_station_cell=dist_station_cell,
            time_station_cell=time_station_cell,
        )
    )

    np.save(os.path.join(output_dir, "distance_cell_cell.npy"), dist_cell_cell)
    np.save(os.path.join(output_dir, "time_cell_cell.npy"), time_cell_cell)
    np.save(
        os.path.join(output_dir, "distance_cell_station_cell.npy"),
        dist_cell_station_cell,
    )
    np.save(
        os.path.join(output_dir, "time_cell_station_cell.npy"), time_cell_station_cell
    )
    np.save(os.path.join(output_dir, "id_cell_station_cell.npy"), id_cell_station_cell)


if __name__ == "__main__":
    create_dist_time_matrices(
        snapping_points_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "snapped_point_from_centroids.shp"
        ),
        petrol_stations_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "petrol_stations.csv"
        ),
        output_dir=config.ZONE_FILES_DIRECTORY,
    )
