import os
from typing import List

import fiona
from shapely.geometry import Polygon, Point, shape

import config


def add_cell_ids_to_snapping_points(
    operator_grid_path: str, zone_grid_path: str, snapping_points_path: str
) -> None:
    """
    Function adds cell ids to snapping points that don't have one.
    :param operator_grid_path: path to shapefile with operator grid
    :param zone_grid_path: path to shapefile with zone grid
    :param snapping_points_path: path to shapefile with snapping points
    :return: None
    """
    with fiona.open(operator_grid_path, "r") as shp:
        all_cells = list(shp)
    with fiona.open(zone_grid_path) as shp:
        grid_feats = list(shp)
        polygon_meta = shp.meta
        grid_ids: List[str] = [f["properties"]["id"] for f in grid_feats]
    with fiona.open(snapping_points_path) as shp:
        snapping_points_feats = list(shp)
        point_meta = shp.meta

    with fiona.open(
        snapping_points_path,
        "w",
        **point_meta,
        encoding="utf-8",
    ) as point_sink:
        with fiona.open(
            zone_grid_path, "a", encoding="utf-8", **polygon_meta
        ) as zone_grid_sink:
            for i, s in enumerate(snapping_points_feats):
                p: Point = shape(s["geometry"])
                if not s["properties"]["id"]:
                    for c in all_cells:
                        g: Polygon = shape(c["geometry"])
                        if g.intersects(p):
                            snapping_points_feats[i]["properties"]["id"] = c[
                                "properties"
                            ]["id"]
                            if c["properties"]["id"] not in grid_ids:
                                zone_grid_sink.write(c)
                            break
        for sp in snapping_points_feats:
            point_sink.write(sp)


if __name__ == "__main__":
    add_cell_ids_to_snapping_points(
        operator_grid_path="",
        zone_grid_path=os.path.join(config.ZONE_FILES_DIRECTORY, "zone_grid.shp"),
        snapping_points_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "snapped_point_from_centroids.shp"
        ),
    )
