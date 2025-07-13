import os
from collections import OrderedDict
from typing import List

import fiona
from fastkml import kml
from pyproj import Transformer
from shapely.geometry import LineString, mapping, Point, MultiPolygon, Polygon
from shapely.ops import transform
from tqdm import tqdm


def convert_roads_line_layer_to_points(
    roads_input_file_path: str,
    zone_kml_path: str,
    output_file_path: str,
    allowed_road_type: List[str],
    distance_between_points: int,
    original_spatial_reference: str,
    local_spatial_reference: str,
) -> None:
    """
    Function creates shapefile with roads line layer converted to point layer.

    :param roads_input_file_path: path to shapefile with roads line layer
    :param zone_kml_path: path to zone kml file
    :param output_file_path: path to save shapefile with roads point layer
    :param allowed_road_type: list of accepted OSM road types
    :param distance_between_points: distance between points in meters
    :param original_spatial_reference: spatial reference of input file
    :param local_spatial_reference: local spatial reference
    :return: None
    """
    transformer = Transformer.from_proj(
        original_spatial_reference, local_spatial_reference, always_xy=True
    )
    inverse_transformer = Transformer.from_proj(
        local_spatial_reference, original_spatial_reference, always_xy=True
    )

    with open(zone_kml_path) as f:
        z = kml.KML()
        z.from_string(f.read())
    zone_geometries: List[Polygon] = []
    for f in z.features():
        for x in f.features():
            zone_geometries.append(transform(transformer.transform, x.geometry))

    zone_geometries: MultiPolygon = MultiPolygon(zone_geometries).buffer(distance=0)

    with fiona.open(roads_input_file_path) as line_roads_file:
        meta = line_roads_file.meta
        driver = meta["driver"]
        crs = meta["crs"]
        vertices_schema = {
            "geometry": "Point",
            "properties": OrderedDict([("lat", "float"), ("lon", "float")]),
        }
        with fiona.open(
            output_file_path,
            "w",
            encoding="utf-8",
            driver=driver,
            crs=crs,
            schema=vertices_schema,
        ) as vertices_roads_file:
            for road in tqdm(list(line_roads_file)):
                if road["properties"]["fclass"] not in allowed_road_type:
                    continue
                coords = list(zip(*(road["geometry"]["coordinates"])))
                road_transformed_coords = transformer.transform(coords[0], coords[1])
                road_transformed = LineString(
                    zip(road_transformed_coords[0], road_transformed_coords[1])
                )
                if road_transformed.intersects(zone_geometries):
                    road_transformed = road_transformed.intersection(zone_geometries)
                    for dist in range(
                        0, int(road_transformed.length), distance_between_points
                    ):
                        vertex = road_transformed.interpolate(dist)
                        vertex_transformed_coords = inverse_transformer.transform(
                            vertex.xy[0], vertex.xy[1]
                        )
                        vertex = Point(
                            vertex_transformed_coords[0][0],
                            vertex_transformed_coords[1][0],
                        )
                        vertices_roads_file.write(
                            {
                                "geometry": mapping(vertex),
                                "properties": OrderedDict(
                                    [("lat", vertex.x), ("lon", vertex.y)]
                                ),
                            }
                        )
        print(f"file with roads as vertices saved at {output_file_path}")


if __name__ == "__main__":
    import config

    in_pth = ""
    out_path = os.path.join(config.ZONE_FILES_DIRECTORY, "roads_points.shp")
    zone_kml_pth = os.path.join(config.ZONE_FILES_DIRECTORY, "zone.kml")
    accepted_road_ftype = [
        "living_street",
        "primary_link",
        "residential",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    ]
    distance_between_road_vertices = 10
    wgs84 = "epsg:4326"
    cs92 = "epsg:2180"

    convert_roads_line_layer_to_points(
        roads_input_file_path=in_pth,
        zone_kml_path=zone_kml_pth,
        output_file_path=out_path,
        allowed_road_type=accepted_road_ftype,
        distance_between_points=distance_between_road_vertices,
        original_spatial_reference=wgs84,
        local_spatial_reference=cs92,
    )
