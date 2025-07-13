import itertools
import os
from typing import Dict, List

import fiona
import numpy as np
import pyproj
import rtree
from shapely.geometry import shape, Polygon
from tqdm import tqdm

import config


def run(
    snapping_points_path: str,
    zone_grid_path: str,
    distance_cell_cell_path: str,
    output_path: str,
    original_crs: str,
    local_crs: str,
    distance_factor: float,
    n_neighbors: int,
) -> None:
    """
    Function creates shapefile with possibly incorrect snapping points (wrong side of the road, placed on highway etc.)
    by comparing Euclidean distance between points and road distance from matrix.
    :param snapping_points_path: path to shapefile with snapping points
    :param zone_grid_path: path to shapefile with zone grid
    :param distance_cell_cell_path: path to numpy matrix with distances cell-cell
    :param output_path: path to shapefile where possibly incorrect snapping points will be saved
    :param original_crs: original crs (ex. epsg:4326)
    :param local_crs: local crs (ex. epsg: 2180)
    :param distance_factor: how many times road distance has to be larger than Euclidean distance to consider point as suspicious
    :param n_neighbors: how many suspicious neighbors have to be to consider point incorrect
    :return: None
    """
    transformation = pyproj.Transformer.from_proj(
        original_crs, local_crs, always_xy=True
    )
    with fiona.open(snapping_points_path, "r") as shp:
        point_feats: List[Dict] = sorted(list(shp), key=lambda f: f["properties"]["id"])
    with fiona.open(zone_grid_path, "r") as shp:
        cell_feats: List[Dict] = sorted(list(shp), key=lambda f: f["properties"]["id"])
    dcc: np.array = np.load(distance_cell_cell_path)
    print("creating euclidean distances")
    snap_euclidean_distances = np.zeros((len(point_feats), len(point_feats)))
    all_relations = list(itertools.combinations(range(len(point_feats)), 2))
    for i, j in tqdm(all_relations):
        p1_lon, p1_lat = transformation.transform(
            *point_feats[i]["geometry"]["coordinates"]
        )
        p2_lon, p2_lat = transformation.transform(
            *point_feats[j]["geometry"]["coordinates"]
        )

        dist_snap = np.sqrt(np.power(p1_lon - p2_lon, 2) + np.power(p1_lat - p2_lat, 2))
        snap_euclidean_distances[i, j] = dist_snap
        snap_euclidean_distances[j, i] = dist_snap

    # create cells spatial index
    sindex = rtree.index.Index(interleaved=True)
    cell_geoms: List[Polygon] = []
    for i, cell in enumerate(cell_feats):
        g: Polygon = shape(cell["geometry"])
        cell_geoms.append(g)
        sindex.insert(i, g.bounds)

    # create adjacency list
    adjacency_list = [[] for _ in range(len(cell_feats))]
    for i, cell in enumerate(cell_feats):
        g = cell_geoms[i]
        candidates = sindex.intersection(g.bounds)
        for idx in candidates:
            if cell_geoms[idx].intersects(g) and i != idx:
                adjacency_list[i].append(idx)

    suspicious_cells = []

    for cell_id in range(len(cell_geoms)):
        neighbors_description = ""
        n_wrong = 0
        for neighbor_id in adjacency_list[cell_id]:
            dist = dcc[cell_id, neighbor_id]
            if dist > distance_factor * snap_euclidean_distances[cell_id, neighbor_id]:
                neighbors_description += (
                    f"i:{cell_feats[neighbor_id]['properties']['id']}, d:{dist}\n"
                )
                n_wrong += 1
        if neighbors_description and n_wrong >= n_neighbors:
            feat = cell_feats[cell_id]
            feat["properties"]["neighbors"] = neighbors_description
            suspicious_cells.append(feat)

    schema = config.POLYGON_SCHEMA
    schema["properties"]["neighbors"] = "str"
    driver = "ESRI Shapefile"
    with fiona.open(output_path, "w", driver=driver, schema=schema) as sink:
        for c in suspicious_cells:
            sink.write(c)
    print(f"Suspicious cells: {len(suspicious_cells)}")


if __name__ == "__main__":
    run(
        snapping_points_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "snapped_point_from_centroids.shp"
        ),
        zone_grid_path=os.path.join(config.ZONE_FILES_DIRECTORY, "zone_grid.shp"),
        distance_cell_cell_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "distance_cell_cell.npy"
        ),
        output_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "potential_wrong_snapping_points.shp"
        ),
        original_crs="epsg:4326",
        local_crs="epsg:2180",
        distance_factor=config.NEIGHBOR_RANGE_FACTOR,
        n_neighbors=config.MIN_WRONG_NEIGHBORS,
    )
