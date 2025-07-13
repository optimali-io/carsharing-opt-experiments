import os
from typing import List

import fiona
import pyproj
import rtree
from fastkml import kml
from shapely.geometry import MultiPolygon, Polygon, Point, shape
from shapely.ops import transform
from tqdm import tqdm

import config


def create_shapefiles(
    kml_path: str,
    roads_path: str,
    gconf_path: str,
    output_dir: str,
    original_crs: str,
    local_crs: str,
    area_threshold: float,
    cells_with_rents: List[str] = [],
) -> None:
    """
    Function creates three shapefiles. Grid with cells definitely in zone, grid with cells possibly in zone
    and snapping points.
    :param kml_path: path of kml file with zone borders
    :param roads_path: path of shapefile with point roads layer
    :param gconf_path: path of shapefile with operator grid (GeoConfigFiles)
    :param output_dir: path of directory where resulting shapefiles will be saved
    :param original_crs: original projection (ex. "epsg:4326")
    :param local_crs: local projection (ex. "epsg:2180")
    :param area_threshold: minimal ratio of (intersection between cell and zone) to (cell area) to consider cell to be definitely in zone
    :param cells_with_rents: List with ids of cells with noted rent activity
    :return: None
    """
    transformation = pyproj.Transformer.from_proj(
        original_crs, local_crs, always_xy=True
    )
    inverse_transformation = pyproj.Transformer.from_proj(
        local_crs, original_crs, always_xy=True
    )
    cells_in_zone: List[Polygon] = []
    cells_maybe_in_zone: List[Polygon] = []
    snapping_points: List[Point] = []

    # Create geometry for zone.
    with open(kml_path) as f:
        z = kml.KML()
        z.from_string(f.read())
    zone_polygons: List[Polygon] = []
    for f in z.features():
        for x in f.features():
            zone_polygons.append(x.geometry)
    zone_geometries: MultiPolygon = MultiPolygon(zone_polygons).buffer(distance=0)
    bounds_before_transform = zone_geometries.bounds
    zone_geometries = transform(transformation.transform, zone_geometries)
    print(f"loading roads point layer")

    sindex = rtree.index.Index(interleaved=True)
    # Create geometries for roads points.
    with fiona.open(roads_path) as shp:
        points_list = list(shp)
        points_geometries = []
        print("creating point geometries and spatial index")
        for i, p in enumerate(tqdm(points_list)):
            g: Point = transform(transformation.transform, shape(p["geometry"]))
            points_geometries.append(g)
            sindex.insert(i, g.bounds)

    # Find cells in zone.
    print("loading cells in zone\n")
    with fiona.open(gconf_path) as shp:
        for feat in tqdm(list(shp.values(bbox=bounds_before_transform))):
            g: Polygon = transform(transformation.transform, shape(feat["geometry"]))
            if zone_geometries.intersects(g):
                candidate_points = sindex.intersection(g.bounds)
                min_dist = 99999
                snapped_point = None
                for idx in candidate_points:
                    p_geom = points_geometries[idx]
                    if p_geom.intersects(g):
                        dist = g.centroid.distance(p_geom)
                        if dist < min_dist:
                            min_dist = dist
                            snapped_point = p_geom
                if snapped_point:
                    snapping_points.append(snapped_point)
                    cells_in_zone.append(feat)
                elif (
                    feat["properties"]["id"] in cells_with_rents
                    or zone_geometries.intersection(g).area / g.area >= area_threshold
                ):
                    cells_maybe_in_zone.append(feat)

    # Save results.
    with fiona.open(
        os.path.join(output_dir, "zone_grid.shp"),
        "w",
        encoding="utf-8",
        driver=config.DRIVER,
        schema=config.POLYGON_SCHEMA,
        crs=original_crs,
    ) as zone_grid:
        with fiona.open(
            os.path.join(output_dir, "snapped_point_from_centroids.shp"),
            "w",
            driver=config.DRIVER,
            encoding="utf-8",
            schema=config.POINT_SCHEMA,
            crs=original_crs,
        ) as snapping_points_file:
            for i, c in enumerate(cells_in_zone):
                zone_grid.write(c)
                cell_id: str = c["properties"]["id"]
                snap_point: Point = snapping_points[i]
                p_coords = snap_point.coords[0]
                transformed_coords = inverse_transformation.transform(*p_coords)
                snapping_points_file.write(
                    {
                        "geometry": {
                            "coordinates": transformed_coords,
                            "type": "Point",
                        },
                        "properties": {"id": cell_id, "address": ""},
                    }
                )
    with fiona.open(
        os.path.join(output_dir, "potential_cells.shp"),
        "w",
        driver=config.DRIVER,
        schema=config.POLYGON_SCHEMA,
        crs=original_crs,
    ) as potential_cells:
        for feat in cells_maybe_in_zone:
            potential_cells.write(feat)


if __name__ == "__main__":
    create_shapefiles(
        kml_path=os.path.join(config.ZONE_FILES_DIRECTORY, "zone.kml"),
        roads_path=os.path.join(config.ZONE_FILES_DIRECTORY, "roads_points.shp"),
        gconf_path="",
        output_dir=config.ZONE_FILES_DIRECTORY,
        original_crs="epsg:4326",
        local_crs="epsg:2180",
        area_threshold=config.INTERSECTION_TO_CELL_AREA_RATIO_THRESHOLD,
        cells_with_rents=[],
    )
