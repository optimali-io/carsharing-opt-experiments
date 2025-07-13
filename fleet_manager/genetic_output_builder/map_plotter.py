import logging
from collections import namedtuple
from typing import Dict, List, Tuple

import folium
import numpy as np
import shapely.geometry
from branca.utilities import color_brewer
from fastkml import KML

from core.utils.id_to_index_mapper import IdToIndexMapper
from fleet_manager.data_model.zone_model import Vehicle, ZoneData

log = logging.getLogger(__name__)


def get_cell_yx(cell_id: str):
    pos = [float(x) for x in cell_id.split("-")]
    return pos[1], pos[0]


class MapPlotter:
    """Class responsible for plotting maps based on zone status and relocations list."""

    def __init__(self, zone_data: ZoneData, zone_kml_file: KML, phenotype: Dict, cell_ids: List[str]):
        self.zone_data: ZoneData = zone_data
        self._zone_kml_file: KML = zone_kml_file
        self.folium_map = None
        self.phenotype: Dict = phenotype
        self.id_index_mapper = IdToIndexMapper(cell_ids, sort=False)

    def run_plotting(self):
        """Execute plotting of all map's elements."""

        lats = sorted([c.latitude for c in self.zone_data.cells])
        lons = sorted([c.longitude for c in self.zone_data.cells])
        center = lats[int(len(lats) / 2)], lons[int(len(lons) / 2)]

        self.folium_map = folium.Map(location=center, zoom_start=11.5, tiles="cartodbpositron")
        # folium.TileLayer("Stamen Toner").add_to(self.folium_map)
        folium.TileLayer("openstreetmap").add_to(self.folium_map)

        self._plot_service_route()
        self._plot_zone()
        self._plot_demand()
        self._plot_gas_stations()
        self._plot_power_stations()
        self._plot_vehicles()

        folium.LayerControl(collapsed=False).add_to(self.folium_map)


    def _plot_zone(self):
        """Plot Zone borders."""
        feature_group = folium.map.FeatureGroup(name="Zone")
        for feature1 in self._zone_kml_file.features:
            for feature2 in feature1.features:
                polygon = feature2.geometry
                m = shapely.geometry.mapping(polygon)
                points = [[p[1], p[0]] for p in m["coordinates"][0]]
                feature_group.add_child(folium.PolyLine(locations=points, color="red"))

        self.folium_map.add_child(feature_group, index=30)

    def _plot_demand(self):
        """Plot demand points per cell."""
        cb = color_brewer("YlOrRd", n=9)

        feature_group = folium.map.FeatureGroup(name="Demand 24h", show=False)

        for cell_id in self.zone_data.zone_cell:
            p = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(cell_id))
            cell_demand = self.zone_data.demand[cell_id]
            cell_color = MapPlotter._get_color(cb, cell_demand)

            popup = MapPlotter._demand_popup(p, cell_id, cell_demand)
            feature_group.add_child(
                folium.Circle(p, radius=230, popup=popup, fill=True, fill_opacity=0.5, color=cell_color, stroke=False)
            )
        self.folium_map.add_child(feature_group, index=100)

    def _plot_service_route(self):
        """Plot route of service teams."""
        for chromosome in self.phenotype["genome"].values():
            su_id = chromosome["service_team_id"]
            team_kind = chromosome["kind"]
            feature_group = folium.map.FeatureGroup(name="Route " + str(su_id))
            service_team = self.zone_data.service_team[su_id]
            prv_pos = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(service_team.start_cell_id))
            prv_cell_id = service_team.start_cell_id

            for i, gene in enumerate(chromosome["genes"]):
                vehicle = self.zone_data.fleet[gene["vehicle"]]

                source_cell_id = gene["source_cell_id"]
                destination_cell_id = gene["destination_cell_id"]
                if team_kind == 1:
                    try:
                        next_vehicle: Vehicle = self.zone_data.fleet[chromosome["genes"][i + 1]["vehicle"]]
                        destination_pos = next_vehicle.latitude, next_vehicle.longitude
                    except IndexError:
                        destination_pos = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(destination_cell_id))
                else:
                    destination_pos = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(destination_cell_id))

                source_pos = vehicle.latitude, vehicle.longitude

                self._draw_line(
                    feature_group, "To Relocation {}".format(i), prv_cell_id, source_cell_id, prv_pos, source_pos, 5
                )
                if gene["action"] == 1:
                    power_station = self.zone_data.gas_station[
                        self.zone_data.id_cell_station_cell[source_cell_id, destination_cell_id]
                    ]
                    station_pos = power_station.latitude, power_station.longitude
                    self._draw_line_gas_station(
                        feature_group,
                        f"Relocation via Gas Station {power_station.address}",
                        source_cell_id,
                        destination_cell_id,
                        source_pos,
                        station_pos,
                        destination_pos,
                        12,
                        "blue",
                    )
                elif gene["action"] == 2:
                    self._draw_line(
                        feature_group,
                        "Relocation {}".format(i),
                        source_cell_id,
                        destination_cell_id,
                        source_pos,
                        destination_pos,
                        12,
                        "green",
                    )
                if gene["action"] == 3:
                    # plugging
                    power_station_id = [
                        p.power_station_id for p in self.zone_data.power_station if p.cell_id == destination_cell_id
                    ][0]
                    power_station = self.zone_data.power_station[power_station_id]
                    station_pos = power_station.latitude, power_station.longitude
                    self._draw_line(
                        feature_group,
                        f"Plugging vehicle at Power Station {power_station.power_station_id}",
                        source_cell_id,
                        destination_cell_id,
                        source_pos,
                        station_pos,
                        12,
                        "orange",
                    )
                if gene["action"] == 4:
                    # unplugging
                    station_pos = vehicle.latitude, vehicle.longitude
                    self._draw_line(
                        feature_group,
                        f"Unplugging vehicle and Relocation to {destination_pos}",
                        source_cell_id,
                        destination_cell_id,
                        station_pos,
                        destination_pos,
                        12,
                        "lightgreen",
                    )
                prv_cell_id = destination_cell_id
                prv_pos = destination_pos

            self._draw_line(
                feature_group,
                "To Depot {}".format(su_id),
                prv_cell_id,
                service_team.end_cell_id,
                prv_pos,
                get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(service_team.end_cell_id)),
                5,
            )

            start_st_point = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(service_team.start_cell_id))
            popup = MapPlotter._base_station_popup(start_st_point)
            base_station = folium.RegularPolygonMarker(
                location=start_st_point, fill_color="grey", number_of_sides=5, radius=10, popup=popup
            )
            feature_group.add_child(base_station)

            # if service_team['start_cell'] != service_team['end_cell']:
            end_st_point = get_cell_yx(self.id_index_mapper.revert_index_to_cell_id(service_team.end_cell_id))
            popup = MapPlotter._base_station_popup(end_st_point)
            base_station = folium.Circle(
                location=end_st_point, fill_color="red", number_of_sides=20, radius=10, popup=popup
            )
            feature_group.add_child(base_station)

            self.folium_map.add_child(feature_group, index=400)

    def _plot_vehicles(self):
        """Plot positions of vehicles. Colors symbolize wait time."""
        feature_group_00_minus = folium.map.FeatureGroup(name="Vehicles Status 0 minus", show=False)
        feature_group_00_12 = folium.map.FeatureGroup(name="Vehicles Status 0 - 12", show=False)
        feature_group_12_24 = folium.map.FeatureGroup(name="Vehicles Status 12 - 24", show=False)
        feature_group_24_48 = folium.map.FeatureGroup(name="Vehicles Status 24 - 48", show=False)
        feature_group_48_plus = folium.map.FeatureGroup(name="Vehicles Status 48 plus", show=False)

        for vehicle in self.zone_data.fleet:
            p = vehicle.latitude, vehicle.longitude
            popup = MapPlotter._vehicle_popup(
                p,
                vehicle.base_cell_id,
                vehicle.wait_time / 60.0,
                vehicle.backend_vehicle_id,
                str(vehicle.backend_vehicle_id),
                vehicle.range * 7 / 100,
            )

            if vehicle.wait_time >= 2880:
                icon_color = "red"
                feature_group = feature_group_48_plus
            elif vehicle.wait_time >= 1440:
                icon_color = "orange"
                feature_group = feature_group_24_48
            elif vehicle.wait_time >= 720:
                icon_color = "yellow"
                feature_group = feature_group_12_24
            elif vehicle.wait_time >= 0:
                icon_color = "lightgreen"
                feature_group = feature_group_00_12
            else:
                icon_color = "white"
                feature_group = feature_group_00_minus

            if vehicle.drive == 1:
                color = "black"
            if 0 < vehicle.range < self.zone_data.revenue_reduction_range and vehicle.drive == 1:
                color = "blue"

            if vehicle.drive == 2:
                color = "green"
                if 0 < vehicle.range < self.zone_data.revenue_reduction_range:
                    color = "red"

            icon = folium.Icon(color=color, icon_color=icon_color, icon="circle", prefix="fa")

            feature_group.add_child(folium.Marker(p, popup=popup, icon=icon))

        self.folium_map.add_child(feature_group_00_minus, index=310)
        self.folium_map.add_child(feature_group_00_12, index=320)
        self.folium_map.add_child(feature_group_12_24, index=330)
        self.folium_map.add_child(feature_group_24_48, index=340)
        self.folium_map.add_child(feature_group_48_plus, index=350)

    def _plot_gas_stations(self):
        """Plot positions of gas stations."""
        feature_group = folium.map.FeatureGroup(name="Petrol Stations", show=False)
        for station in self.zone_data.gas_station:
            station_pos = station.latitude, station.longitude
            cell_id = station.cell_id
            popup = MapPlotter._gas_station_popup(station_pos, cell_id)
            ps_icon = folium.Icon(color="darkblue", icon_color="white", icon="tint")
            feature_group.add_child(folium.Marker(station_pos, popup=popup, icon=ps_icon))

        self.folium_map.add_child(feature_group, index=250)

    def _plot_power_stations(self):
        """Plot positions of power stations."""
        feature_group = folium.map.FeatureGroup(name="Power Stations", show=False)
        for station in self.zone_data.power_station:
            station_pos = station.latitude, station.longitude
            cell_id = station.cell_id
            popup = MapPlotter._power_station_popup(station_pos, cell_id)
            ps_icon = folium.Icon(color="darkgreen", icon_color="white", icon="tint")
            feature_group.add_child(folium.Marker(station_pos, popup=popup, icon=ps_icon))

        self.folium_map.add_child(feature_group, index=250)

    def _draw_line(
        self,
        feature_group: folium.map.FeatureGroup,
        popup_name: str,
        cell_id_1: int,
        cell_id_2: int,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
        weight: int,
        color: str = "grey",
    ):
        """
        Draw line between two points.
        :param feature_group: group of features for the map.
        :param popup_name: name of the popup
        :param cell_id_1: id (matrix index) of the source cell
        :param cell_id_2: id (matrix index) of the destination cell
        :param pos1: tuple (Lat, Lon), source position
        :param pos2: tuple (Lat, Lon), destination position
        :param weight: weight of the feature
        """
        if cell_id_1 != cell_id_2:
            distance = self.zone_data.distance_cell_cell[cell_id_1, cell_id_2]
            popup = MapPlotter._route_popup(popup_name, pos1, pos2, distance)
            feature_group.add_child(
                folium.PolyLine(locations=[pos1, pos2], color=color, weight=weight, opacity=0.7, popup=popup)
            )
            arrow = MapPlotter._get_arrow(locations=[pos1, pos2], color="blue", popup=popup)
            feature_group.add_child(arrow)

    def _draw_line_gas_station(
        self,
        feature_group: folium.map.FeatureGroup,
        popup_name: str,
        c_id_source: int,
        c_id_destination: int,
        pos_source: Tuple[float, float],
        pos_gas_station: Tuple[float, float],
        pos_destination: Tuple[float, float],
        weight: int,
        color="grey",
    ):
        """
        Draw line between two points.
        :param feature_group: group of features for the map.
        :param popup_name: name of the popup
        :param c_id_source: id (matrix index) of the source cell
        :param c_id_destination: id (matrix index) of the destination cell
        :param pos_source: tuple (Lat, Lon), source position
        :param pos_gas_station: tuple (Lat, Lon), gas station position
        :param pos_destination: tuple (Lat, Lon), destination position
        :param weight: weight of the feature
        """
        distance = self.zone_data.distance_cell_station_cell[c_id_source, c_id_destination]
        popup = MapPlotter._route_popup(popup_name, pos_source, pos_destination, distance)
        feature_group.add_child(
            folium.PolyLine(
                locations=[pos_source, pos_gas_station, pos_destination],
                color=color,
                weight=weight,
                opacity=0.7,
                popup=popup,
            )
        )
        arrow = MapPlotter._get_arrow(locations=[pos_source, pos_gas_station], color="blue", popup=popup)
        feature_group.add_child(arrow)
        arrow = MapPlotter._get_arrow(locations=[pos_gas_station, pos_destination], color="blue", popup=popup)
        feature_group.add_child(arrow)

    @staticmethod
    def _route_popup(name: str, p1: Tuple[float, float], p2: Tuple[float, float], distance: int):
        """
        Create popup of the service team route.
        :param name: name of the popup
        :param p1: (Lat, Lon) of the source position
        :param p2: (Lat, Lon) of the destination position
        :param distance: distance of the route
        """
        r_link = MapPlotter._route_link(p1, p2, "Route")
        p1_link = MapPlotter._place_link(p1)
        p2_link = MapPlotter._place_link(p2)
        popup = MapPlotter._table(
            name,
            "Distance",
            "{:.2f}km".format(distance / 1000),
            "Route",
            r_link,
            "Source",
            p1_link,
            "Destination",
            p2_link,
        )
        return popup

    @staticmethod
    def _vehicle_popup(
        position: Tuple[float, float], cell_id: int, hours: float, vehicle_id: int, vehicle_name: str, fuel: float
    ):
        """
        Create popup of the vehicle position.
        :param position: (Lat, Lon) of the vehicle
        :param cell_id: id (index in matrix) of the cell
        :param hours: wait time of the vehicle (h)
        :param vehicle_id:  id of the vehicle (backend id)
        :param vehicle_name: name (foreign id) of the vehicle
        :param fuel: fuel level (l) of the vehicle
        """
        link = MapPlotter._place_link(position)
        popup = MapPlotter._table(
            "Car Status",
            "CellID",
            cell_id,
            "Hours",
            hours,
            "Car",
            "{}, {}".format(vehicle_id, vehicle_name),
            "Fuel",
            fuel,
            "Lat/Lon",
            link,
        )
        return popup

    @staticmethod
    def _base_station_popup(pos: Tuple[float, float]):
        """
        Create popup of the base station.
        :param pos: position of the base station
        """
        link = MapPlotter._place_link(pos)
        popup = MapPlotter._table("Depot", "Lat/Lon", link)
        return popup

    @staticmethod
    def _get_arrow(
        locations: List[Tuple[float, float]], color: str = "blue", size: int = 8, popup: folium.map.Popup = None
    ):
        """
        Create arrow marker for provided positions.
        :param locations: list of points
        :param color: guess
        :param size: size of the arrow
        :param popup: popup for the arrow
        """
        Point = namedtuple("Point", field_names=["lat", "lon"])

        p1 = Point(locations[0][0], locations[0][1])
        p2 = Point(locations[1][0], locations[1][1])

        rotation = MapPlotter._get_bearing(p1, p2) - 90

        arrow_lat = p1[0] + (p2[0] - p1[0]) * 0.25
        arrow_lon = p1[1] + (p2[1] - p1[1]) * 0.25
        arrow = folium.RegularPolygonMarker(
            location=[arrow_lat, arrow_lon],
            fill_color=color,
            number_of_sides=3,
            radius=size,
            rotation=rotation,
            popup=popup,
        )
        return arrow

    @staticmethod
    def _get_bearing(pos1: Tuple[float, float], pos2: Tuple[float, float]):
        """
        Get bearing of the line between two positions.
        :param pos1: (Lat, Lon) source position
        :param pos2: (Lat, Lon) destination position
        """
        long_diff = np.radians(pos2.lon - pos1.lon)

        lat1 = np.radians(pos1.lat)
        lat2 = np.radians(pos2.lat)

        x = np.sin(long_diff) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(long_diff))
        bearing = np.degrees(np.arctan2(x, y))

        # adjusting for compass bearing
        if bearing < 0:
            return bearing + 360
        return bearing

    @staticmethod
    def _demand_popup(pos: Tuple[float, float], cell_id: int, cell_demand: float):
        """
        Create popup for the demand point.
        :param pos: (Lat, Lon) position of the demand
        :param cell_id: id (index in matrix) of the demand
        :param cell_demand: value of the demand
        """
        link = MapPlotter._place_link(pos)
        popup = MapPlotter._table(
            "Cell Demand", "CellId", cell_id, "Demand", "{:.2f}".format(cell_demand), "Lat/Lon", link
        )
        return popup

    @staticmethod
    def _table(name, *args):
        """Create the html table for provided arguments."""
        table = "<table>"
        table += '<tr><th colspan="3">{}</th></tr>'.format(name)
        for i in range(0, len(args), 2):
            row_name = args[i]
            row_value = args[i + 1]
            table += "<tr><th>{}</th><td>&nbsp;&nbsp;&nbsp;</td><td>{}</td></tr>".format(row_name, row_value)
        table += "</table>"
        return table

    @staticmethod
    def _route_link(source_pos: Tuple[float, float], destintion_pos: Tuple[float, float], name: str):
        """
        Create the link for the route.
        :param source_pos: (Lat, Lon) of the source position
        :param destintion_pos: (Lat, Lon) of the destination position
        """
        slat, slon = source_pos
        dlat, dlon = destintion_pos
        return (
            "<a"
            ' target="_blank"'
            ' href="https://maps.google.com?saddr={:.6f},{:.6f}&daddr={:.6f},{:.6f}">'
            "{}</a>".format(slat, slon, dlat, dlon, name)
        )

    @staticmethod
    def _place_link(pos: Tuple[float, float], name: str = None):
        """
        Create the link for the place.
        :param pos: (Lat, Lon) position of the place
        :param name: name of the place
        """
        lat, lon = pos
        if name is not None:
            return '<a target="_blank" href="https://www.google.com/maps/place/{:.6f},{:.6f}">{}</a>'.format(
                lat, lon, name
            )
        else:
            return '<a target="_blank" href="https://www.google.com/maps/place/{:.6f},{:.6f}">{:.6f},{:.6f}</a>'.format(
                lat,
                lon,
                lat,
                lon,
            )

    @staticmethod
    def _gas_station_popup(pos: Tuple[float, float], cell_id: int):
        """
        Create popup for the gas station.
        :param pos: (Lat, Lon) position of the gas station
        :param cell_id: id (index in matrix) of the cell
        """
        link = MapPlotter._place_link(pos)
        popup = MapPlotter._table("Petrol Station ", "CellID", cell_id, "Lat/Lon", link)
        return popup

    @staticmethod
    def _power_station_popup(pos: Tuple[float, float], cell_id: int):
        """
        Create popup for the power station.
        :param pos: (Lat, Lon) position of the power station
        :param cell_id: id (index in matrix) of the cell
        """
        link = MapPlotter._place_link(pos)
        popup = MapPlotter._table("Power Station ", "CellID", cell_id, "Lat/Lon", link)
        return popup

    @staticmethod
    def _get_color(cb: color_brewer, cell_demand: float):
        """
        Get color by demand in cell.
        :param cb: color brewer object
        :param cell_demand: demand in the cell
        """
        if cell_demand < 0.2:
            return cb[0]
        elif cell_demand < 1:
            return cb[1]
        else:
            idx = min(int(len(cb) * cell_demand / 15) + 2, len(cb) - 1)
            return cb[idx]
