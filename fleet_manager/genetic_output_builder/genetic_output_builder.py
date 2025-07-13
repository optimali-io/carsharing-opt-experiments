import logging
from datetime import date
from typing import Any, Dict, List

from core.utils.values_formatter import str6
from fleet_manager.data_model.genetic_model import Action, Gene, Phenotype
from fleet_manager.data_model.zone_model import GasStation, PowerStation, ZoneData

log = logging.getLogger(__name__)


class GeneticOutputBuilder:
    """
    Creates vehicle actions list from phenotype.
    """

    def __init__(self):
        self.version = None
        self.file_name_timestamp_str: str = None
        self.zone_data: ZoneData = None
        self.vehicles_actions: List[Action] = None
        self.phenotype: Phenotype = None

    def initialize(self, zone_data: ZoneData, phenotype: Phenotype) -> None:
        """
        Initialize GeneticOutputBuilder.

        :param zone_data: zone_data
        :param phenotype: phenotype
        """

        self.version = None
        self.file_name_timestamp_str: str = str(date.today())
        self.zone_data: ZoneData = zone_data
        self.vehicles_actions: List[Action] = []
        self.phenotype: Phenotype = phenotype

        for chromosome in self.phenotype.genome:
            order: int = 1
            team_kind: int = chromosome.kind
            for i, gene in enumerate(chromosome):
                vehicle = self.zone_data.fleet[gene.vehicle_id]
                vehicle_backend_id = vehicle.backend_vehicle_id
                vehicle_foreign_id = vehicle.foreign_vehicle_id
                vehicle_latitude = vehicle.latitude
                vehicle_longitude = vehicle.longitude
                vehicle_address = vehicle.address
                source_info = self._get_source_details(vehicle_latitude, vehicle_longitude, vehicle_address)
                if team_kind == 1:
                    try:
                        next_vehicle = self.zone_data.fleet[chromosome[i + 1].vehicle_id]
                        destination_info = self._get_destination_details(next_vehicle.latitude, next_vehicle.longitude)
                    except IndexError:
                        destination_position = self.zone_data.get_lat_lon(gene.destination_cell_id)
                        destination_info = self._get_destination_details(*destination_position)
                else:
                    destination_position = self.zone_data.get_lat_lon(gene.destination_cell_id)
                    destination_info = self._get_destination_details(*destination_position)
                if gene.action_type == 1:
                    station_info = self._get_station_details(gene)
                    self._add_vehicles_relocations(
                        Action(
                            vehicle_backend_id,
                            vehicle_foreign_id,
                            zone_data.get_service_team_id_by_index(chromosome.service_team_index),
                            order,
                            source_info,
                            destination_info,
                            station_info,
                        )
                    )
                elif gene.action_type == 2:
                    self._add_vehicles_relocations(
                        Action(
                            vehicle_backend_id,
                            vehicle_foreign_id,
                            zone_data.get_service_team_id_by_index(chromosome.service_team_index),
                            order,
                            source_info,
                            destination_info,
                        )
                    )
                elif gene.action_type == 3:
                    # plugging
                    station_details = self._get_power_station_details(gene)
                    self._add_vehicles_relocations(
                        Action(
                            vehicle_backend_id,
                            vehicle_foreign_id,
                            zone_data.get_service_team_id_by_index(chromosome.service_team_index),
                            order,
                            source_info,
                            destination_info,
                            station_details,
                        )
                    )
                elif gene.action_type == 4:
                    # unplugging
                    self._add_vehicles_relocations(
                        Action(
                            vehicle_backend_id,
                            vehicle_foreign_id,
                            zone_data.get_service_team_id_by_index(chromosome.service_team_index),
                            order,
                            source_info,
                            destination_info,
                            self._get_power_station_details(gene),
                        )
                    )
                order += 1

        self.relocation_summary: dict = self._create_relocation_summary()

    def _create_relocation_summary(self) -> Dict[str, Any]:
        """
        Create relocation summary dictionary.

        :return: relocation summary dictionary
        """
        summary_dict = {
            "version": str(self.version),
            "processed_vehicles_number": str(len(self.zone_data.fleet)),
            "base_revenue": self.phenotype.base_revenue,
            "all_relocations": str(len(self.vehicles_actions)),
            "via_relocations": str(sum(1 for action in self.vehicles_actions if action.station_details)),
            "optimized_revenue": self.phenotype.base_revenue + self.phenotype.diff_revenue,
        }
        return summary_dict

    def _add_vehicles_relocations(self, vehicle_relocation: Action) -> None:
        """
        Append vehicle relocation action to vehicles actions list.

        :param vehicle_relocation: vehicle relocation action
        """

        self.vehicles_actions.append(vehicle_relocation)

    def _get_station_details(self, gene: Gene) -> Dict[str, Any]:
        """
        Create station details dictionary.

        :param gene: gene

        :return: station details dictionary
        """

        gas_station: GasStation = self.zone_data.gas_station[
            self.zone_data.id_cell_station_cell[gene.source_cell_id, gene.destination_cell_id]
        ]
        station_info_dict = {
            "station_cell_id": gas_station.cell_id,
            "station_lat": gas_station.latitude,
            "station_lon": gas_station.longitude,
            "station_address": gas_station.address,
            "station_name": gas_station.gas_station_name,
            "station_GoogleMapsLink": self._add_hyperlink(gas_station.latitude, gas_station.longitude),
        }
        return station_info_dict

    def _get_power_station_details(self, gene: Gene) -> Dict[str, Any]:
        """
        Create power station details dictionary.

        :param gene: gene

        :return: station details dictionary
        """

        power_station: PowerStation = self.zone_data.power_station[
            self.zone_data.fleet[gene.vehicle_id].power_station_id
        ]
        station_info_dict = {
            "station_cell_id": power_station.cell_id,
            "station_lat": power_station.latitude,
            "station_lon": power_station.longitude,
            "station_name": f"power station {power_station.power_station_id}",
            "station_GoogleMapsLink": self._add_hyperlink(power_station.latitude, power_station.longitude),
            "station_address": "station address",
        }
        return station_info_dict

    def _get_destination_details(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Create destination details dictionary.

        :param lat: latitude
        :param lon: longitude

        :return: destination details dictionary
        """

        cell_info_dict = {
            f"destination_lat": float(lat),
            f"destination_lon": float(lon),
            f"destination_name": "",
            f"destination_GoogleMapsLink": self._add_hyperlink(lat, lon),
        }
        return cell_info_dict

    def _get_source_details(self, lat: float, lon: float, address: str) -> Dict[str, Any]:
        """
        Create source details dictionary.

        :param lat: latitude
        :param lon: longitude

        :return: source details dictionary
        """

        info_dict = {
            f"source_lat": float(lat),
            f"source_lon": float(lon),
            f"source_address": address,
            f"source_GoogleMapsLink": self._add_hyperlink(lat, lon),
        }
        return info_dict

    @staticmethod
    def _add_hyperlink(latitude: float, longitude: float) -> str:
        """
        Convert a location described by geographic coordinates to a google maps place url.

        :param latitude: latitude
        :param longitude: longitude

        :return: google maps place url
        """

        google_maps_link = r"https://www.google.com/maps/place/"
        hyperlink_string = "{}{},{}".format(google_maps_link, str6(latitude), str6(longitude))
        return hyperlink_string
