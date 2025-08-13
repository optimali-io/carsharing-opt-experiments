Prepare client data
===================

This project is a bunch of scripts necessary to build ZoneFiles.

Before we start:

``cp zone_creator_config.py.example zone_creator_config.py`` <-- fill up


Ok, scripts should work now.

What do you need?
-----------------

- kml file with the zone's shape 
- Shapefile with the operator's grid polygons, same as `<MAIN_PATH>/data/reference_grid`
- csv file with petrol stations
- Here Api developer key
- Some kind of GIS software, for example QGIS 

See example files in `<MAIN_PATH>/data/zones/lodz_synthetic` directory.

What do you have to do?
-----------------------

Fill zone_creator_config.py file. All files will be saved at ZONE_CONFIG_DIRECTORY.
Move operators zone kml file to ZONE_CONFIG_DIRECTORY and name it 'zone.kml'.


0 - Create roads point layer
------------------------
Download zip with OSM shapefiles from geofabrik.de 

(ex. http://download.geofabrik.de/europe/poland/lodzkie-latest-free.shp.zip)

You may open file "gis_osm_roads_free_1.shp" in QGIS and crop roads layer to preferred
size. This is not really essential, but we don't need to process a whole region.
Save file!

Open script "prepare_zone/prepare_zone_0_create_roads.py" and modify parameters in main method.

Example:
```python
if __name__ == '__main__':
   import config
   in_pth = 'path/to/roads/line/layer/data/gis_osm_roads_free_1.shp'
   out_path = os.path.join(config.ZONE_FILES_DIRECTORY,'roads_points.shp')
   zone_kml_pth = os.path.join(config.ZONE_FILES_DIRECTORY, 'zone.kml')
   accepted_road_ftype = ['living_street', 'primary_link', 'residential', 'secondary', 'secondary_link',
                        'tertiary', 'tertiary_link', ]
   distance_between_road_vertices = 10
   wgs84 = 'epsg:4326'
   cs92 = 'epsg:2180'
   
   convert_roads_line_layer_to_points(roads_input_file_path=in_pth,
                                    zone_kml_path=zone_kml_pth,
                                    output_file_path=out_path,
                                    allowed_road_type=accepted_road_ftype,
                                    distance_between_points=distance_between_road_vertices,
                                    original_spatial_reference=wgs84,
                                    local_spatial_reference=cs92)
```

Run script!
Shapefiles with roads point layer will be saved at /path/from/ZONE_CONFIG_DIRECTORY/roads_points.*

1 - Create zone grid and snapping points
------------------------------------
Open script "prepare_zone/prepare_zone_1_make_zone.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    create_shapefiles(kml_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'zone.kml'),
                      roads_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'roads_points.shp'),
                      gconf_path='/path/to/GeoConfigFiles/operator_grid.shp',
                      output_dir=config.ZONE_FILES_DIRECTORY,
                      original_crs='epsg:4326',
                      local_crs='epsg:2180',
                      area_threshold=config.INTERSECTION_TO_CELL_AREA_RATIO_THRESHOLD,
                      cells_with_rents=[])
```
Run script!
This script creates three shapefiles in the ZONE_FILES_DIRECTORY:
- **zone_grid.shp**: File with cells from operator grid that are definitely in the zone. 
  (ratio of area of their intersection with zone borders to cell size is larger than "area_threshold" 
  parameter and cell have a snapping point).
- **snapped_point_from_centroids.shp**: File with snapping points for cells that are definitely in the zone.
- **potential_cells.shp**: File with cells that don't have snapping point, but they have intersection area 
  large enough, or their id was in list of cells used in rents ("cells_with_rents" parameter).
  
Add missing snapping points
---------------------------
Open four layers in QGIS:
1. **zone_grid.shp**
2. **snapped_point_from_centroids.shp**
3. **potential_cells.shp**
4. **zone.kml**


Analyse **potential_cells**, for each cell that you think should be in the zone, add snapping point to
**snapped_point_from_centroids** layer. Don't enter address or id, let next scripts handle it.
Save edits on **snapped_point_from_centroids** layer!

2 - Add cell ids to manually added snapping points
----------------------------------------------
Open script "prepare_zone/prepare_zone_2_fix_zone.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    add_cell_ids_to_snapping_points(
        operator_grid_path='/path/to/GeoConfigFiles/operator.shp',
        zone_grid_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'zone_grid.shp'),
        snapping_points_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'snapped_point_from_centroids.shp')
```
Run script!

This script adds cell id to every snapping point that don't have one.

3 - Add address to each snapping point
----------------------------------
Open script "prepare_zone/prepare_zone_3_add_addresses.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    add_addresses_to_snapping_points(
        snapping_points_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'snapped_point_from_centroids.shp'))
```
Run script!
This script adds address to every snapping point that don't have one.

4 - Download distance and time matrices
-----------------------------------
Open script "prepare_zone/prepare_zone_4_get_dist_time_matrices.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    create_dist_time_matrices(snapping_points_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'snapped_point_from_centroids.shp'),
                              petrol_stations_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'petrol_stations.csv'),
                              output_dir=config.ZONE_FILES_DIRECTORY,
```
Run script!

Script creates distance-cell-cell, time-cell-cell, distance-cell-station-cell, time-cell-station-cell and
    id-cell-station-cell matrices and saves them in "output_dir" directory.

5 - Clean work directory
--------------------
Open script "prepare_zone/prepare_zone_5_clean_temp_files.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    clean_temp_files(work_dir_path=config.ZONE_FILES_DIRECTORY)
```
Run script!

Script removes potential cells and roads points files from a directory.


6 - Check if snapping points are alright
-------------------------------------

Open script "prepare_zone/prepare_zone_6_check_snapping_points.py" and modify parameters in main method.
Example:
```python
if __name__ == '__main__':
    run(snapping_points_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'snapped_point_from_centroids.shp'),
        zone_grid_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'zone_grid.shp'),
        distance_cell_cell_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'distance_cell_cell.npy'),
        output_path=os.path.join(config.ZONE_FILES_DIRECTORY, 'potential_wrong_snapping_points.shp'),
        original_crs='epsg:4326',
        local_crs='epsg:2180',
        distance_factor=config.NEIGHBOR_RANGE_FACTOR,
        n_neighbors=config.MIN_WRONG_NEIGHBORS
        )
```
Run script!

Script checks if here distances are not too far off from Euclidean distances and creates a shapefile 
with possibly incorrect snapping points (wrong side of the road, placed on a highway etc.).

Open this shapefile in QGIS. Check snapping points under suspicious cells. If points are not in the 
place you want them to be, move them (delete bad points and add new points). Don't add cell id and address.
If you moved any points, **go back to "2 - Add cell ids to manually added snapping points"** and proceed
from there.

If not, just clean directory with script **5 - Clean work directory**.

All files ready!
---------------------------------------------------
