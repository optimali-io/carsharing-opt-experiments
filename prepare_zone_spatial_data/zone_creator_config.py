ZONE_FILES_DIRECTORY = ""

HERE_API_KEY = ""

INTERSECTION_TO_CELL_AREA_RATIO_THRESHOLD = 0.0

POLYGON_SCHEMA = {
    'geometry': 'Polygon',
    'properties': {
        'id': 'str'
    }
}

POINT_SCHEMA = {
    'geometry': 'Point',
    'properties': {
        'id': 'str',
        'address': 'str'
    }
}

DRIVER = 'ESRI Shapefile'