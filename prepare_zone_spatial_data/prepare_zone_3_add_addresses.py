import copy
import os
import re
from typing import List, Dict

import fiona
from geopy.geocoders import Here
from tqdm import tqdm

import config


def add_addresses_to_snapping_points(snapping_points_path: str) -> None:
    """
    Function add addresses to snapping points that don't have one.
    :param snapping_points_path: path to shapefile with snapping points.
    :return: None
    """
    geolocator = Here(
        apikey=config.HERE_API_KEY, user_agent="my-applications", timeout=2500
    )
    zip_code_pattern = "\A[0-9][0-9]-[0-9][0-9][0-9]"
    delta = 0.0002  # If we get postal code and no address we move point for a couple of meters up to "max_tries" times.
    max_tries = 5
    with fiona.open(snapping_points_path) as shp:
        snapping_points_feats: List[Dict] = list(shp)
        point_meta: Dict[str, str] = shp.meta
    bad_addresses = 0

    with fiona.open(
        snapping_points_path, "w", encoding="utf-8", **point_meta
    ) as point_sink:
        for f in tqdm(snapping_points_feats):
            if not f["properties"]["address"]:
                lat, lon = f["geometry"]["coordinates"][::-1]
                location = geolocator.reverse(f"{lat},{lon}")
                orig_location = copy.deepcopy(location)
                new_lat, new_lon = lat, lon
                for i in range(max_tries):
                    address = location.address
                    wrong_address = len(re.findall(zip_code_pattern, address)) > 0
                    if wrong_address:
                        new_lat += delta
                        new_lon -= delta
                        location = geolocator.reverse(f"{new_lat},{new_lon}")
                    else:
                        break
                    if i == max_tries - 1:
                        location = orig_location
                        bad_addresses += 1
                f["properties"]["address"] = location.address
            point_sink.write(f)
    print(f"bad addresses: {bad_addresses}")


if __name__ == "__main__":
    add_addresses_to_snapping_points(
        snapping_points_path=os.path.join(
            config.ZONE_FILES_DIRECTORY, "snapped_point_from_centroids.shp"
        )
    )
