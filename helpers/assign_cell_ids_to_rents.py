import os

import pandas as pd
import rtree

from config import settings
from core.db.raw_data_access import get_cell_id_by_coords


def assign_cell_ids(zone_id: str):
    """
    Assigns cell IDs to rents based on their start and end coordinates.
    """
    pth = f"{settings.DATA_DIR}/rents/{zone_id}/"
    files = [os.path.join(pth, f) for f in os.listdir(pth) if f.endswith('.csv')]
    spatial_index = rtree.Index(settings.SPATIAL_INDEX_PATH)
    for f in files:
        rents = pd.read_csv(f, sep=';')
        start_cell_ids = [
            get_cell_id_by_coords(lon=row.start_lon, lat=row.start_lat, index=spatial_index)
            for row in rents.itertuples(index=False)
        ]
        end_cell_ids = [
            get_cell_id_by_coords(lon=row.end_lon, lat=row.end_lat, index=spatial_index)
            for row in rents.itertuples(index=False)
        ]

        rents["start_cell_id"] = start_cell_ids
        rents["end_cell_id"] = end_cell_ids
        rents.to_csv(f, index=False, sep=';')


if __name__ == "__main__":
    zone_id = ""
    if not zone_id:
        raise ValueError("Please provide a valid zone_id.")
    assign_cell_ids(zone_id)