import logging
import time

import numpy as np
import pyximport

pyximport.install()

from core.utils.nptools import assert_ndarray
from fleet_manager.fastsimulator._fast_simulator import fast_simulate_day
from fleet_manager.fastsimulator.demand_realization import distribute_demand_decimal_part
from fleet_manager.fastsimulator.direction_realization import generate_direction_realizations

log = logging.getLogger(__name__)

SEED = int(time.time()) % 375445791


class FastSimulator:
    """
    FastSimulator is a facade that encapsulates all concerns regarding simulation. It utilities Cython routines and
    exposes them in more pythonic way. Usage of FastSimulator consist of two steps. First step is a preparation step
    where simulator internal data structures are build. Second step is a simulation step where vehicles are moved around
    the board according to probabilities provided in preparation step.

    Naming convention for documenting ndarray axis is: H hours, C cells, R rows, V vehicles.
    """

    def __init__(self) -> None:
        """
        Default constructor

        :rtype: None
        """
        self._simulate_in: SimulateIn = None

    def cells_nbr(self) -> int:
        """
        Returns number of cells. Can't be called before prepare.

        :return: number of cells
        :rtype: int
        """
        return self._simulate_in.distance_hour_cell_cell.shape[2]

    def prepare(
        self,
        demand_hc: np.ndarray,
        direction_hcc: np.ndarray,
        distance_hcc: np.ndarray,
        time_hcc: np.ndarray,
        simulations_nbr: int,
    ) -> None:
        """
        Computes simulator internal data.

        :param demand_hc: Rent demand array
        :type demand_hc: numpy.ndarray, shape=(H, C)

        :param direction_hcc: Rent direction array
        :type direction_hcc: numpy.ndarray, shape=(H, C, C)

        :param distance_hcc: Rent distance array
        :type distance_hcc: numpy.ndarray, shape=(H, C, C)

        :param time_hcc: Rent time array
        :type time_hcc: numpy.ndarray, shape=(H, C, C)

        :param simulations_nbr: Number of simulations for each row of vehicles
        :type simulations_nbr: int

        :return: nothing
        :rtype: None
        """

        hours_nbr, cells_nbr = demand_hc.shape

        assert hours_nbr > 0
        assert cells_nbr > 0
        assert simulations_nbr > 0
        assert_ndarray(demand_hc, (hours_nbr, cells_nbr), np.float32, "demand_hc")
        assert_ndarray(direction_hcc, (hours_nbr, cells_nbr, cells_nbr), np.float32, "direction_hcc")
        assert_ndarray(distance_hcc, (hours_nbr, cells_nbr, cells_nbr), np.uint16, "distance_hcc")
        assert_ndarray(time_hcc, (hours_nbr, cells_nbr, cells_nbr), np.uint16, "time_hcc")

        si = SimulateIn()

        si.simulations_nbr = simulations_nbr

        si.region_cell = np.zeros(cells_nbr, dtype=np.uint16)

        si.demand_sample_hour_cell = distribute_demand_decimal_part(demand_hc)
        si.destination_hour_cell_sample = generate_direction_realizations(direction_hcc)

        si.distance_hour_cell_cell = distance_hcc
        si.time_hour_cell_cell = time_hcc

        self._simulate_in = si

    def simulate(self, location_rv: np.ndarray, trace: bool = False) -> "SimulateOut":
        """
        Simulates vehicles movement.

        :param location_rv: Initial location of vehicles. Array contains cell indexes. First array index is a row idx (row of vehicles), second array index is a vehicle idx.
        :type location_rv: numpy.ndarray, shape=(R, V)

        :param trace: If true simulator records intermediate simulation states
        :type trace: bool

        :return: simulation result
        :rtype: SimulateOut
        """

        self._simulate_in.location_row_vehicle = location_rv
        so = _simulate_day(self._simulate_in, trace)
        return so


class SimulateIn:
    """
    SimulateIn contains all input data for Cython _fast_simulator module.
    """

    def __init__(self) -> None:
        """
        Default constructor

        :rtype: None
        """

        self.location_row_vehicle: np.ndarray = None
        """location(cell idx) of vehicle, two dimensional array (R, V)"""
        self.simulations_nbr: int = None
        """number of simulations for each row of locations of vehicles"""
        self.region_cell: np.ndarray = None
        """region idx the cell is in (C)"""
        self.demand_sample_hour_cell: np.ndarray = None
        """rent demand realizations samples array (S, H, C)"""
        self.destination_hour_cell_sample: np.ndarray = None
        """rent direction realizations samples array (H, C, S)"""
        self.distance_hour_cell_cell: np.ndarray = None
        """rent distance array (H, C, C)"""
        self.time_hour_cell_cell: np.ndarray = None
        """rent time array (H, C, C)"""


class SimulateOut:
    """
    SimulateOut contains output data for Cython _fast_simulator module.
    """

    def __init__(self) -> None:
        """
        Default constructor

        :rtype: None
        """

        # in
        self.location_row_vehicle: np.ndarray = None
        """location(cell idx) of vehicle, two dimensional array (R, V)"""
        # out
        self.revenue_row_simulation: np.ndarray = None
        """revenue per row and simulation (R, S)"""
        self.rents_row_simulation: np.ndarray = None
        """rents per row and simulation (R, S)"""
        self.satisfied_demand_row_simulation_region: np.ndarray = None
        """satisfied demand prer row simulation and region (R, S, R)"""
        # trace
        self.revenue_row_simulation_vehicle_hour: np.ndarray = None
        """revenue per row simulation vehicle and hour (R, S, V, H)"""
        self.location_row_simulation_vehicle_hour: np.ndarray = None
        """vehicle location(cell idx) per row simulation vehicle and hour (R, S, V, H)"""
        self.distance_row_simulation_vehicle_hour: np.ndarray = None
        """rent distance per row simulation vehicle and hour (R, S, V, H)"""
        self.time_row_simulation_vehicle_hour: np.ndarray = None
        """rent time per row simulation vehicle and hour (R, S, V, H)"""
        self.rents_row_simulation_vehicle_hour: np.ndarray = None
        """rent per row simulation vehicle and hour (R, S, V, H)"""

    def build_revenue_row_vehicle(self) -> np.ndarray:
        """
        Build average revenue for each vehicle in each row.

        :return: average revenue for each vehicle in each row
        :rtype: numpy.ndarray, shape=(R, V)
        """

        if self.revenue_row_simulation_vehicle_hour is None:
            raise ValueError("SimulateOut object has no trace data, which are necessary to build_revenue_row_vehicle")
        rows_nbr, simulations_nbr, vehicles_nbr, hours_nbr = self.revenue_row_simulation_vehicle_hour.shape
        result_rsv = np.zeros(
            (rows_nbr, simulations_nbr, vehicles_nbr), dtype=self.revenue_row_simulation_vehicle_hour.dtype
        )

        for ridx in range(rows_nbr):
            for sidx in range(simulations_nbr):
                if sidx % 20 == 0:
                    log.info(f"sample {sidx}/{simulations_nbr}")
                a1 = self.location_row_vehicle[ridx].reshape(-1, 1)
                a2 = self.location_row_simulation_vehicle_hour[ridx, sidx]
                start_location_car_hour = np.hstack((a1, a2))
                distributed_revenue_vh = np.zeros(
                    (vehicles_nbr, hours_nbr), dtype=self.revenue_row_simulation_vehicle_hour.dtype
                )
                for hidx in reversed(range(hours_nbr)):
                    location_c = start_location_car_hour[:, hidx]
                    revenue_c = self.revenue_row_simulation_vehicle_hour[ridx, sidx, :, hidx]
                    u, indeces = np.unique(location_c, return_inverse=True)
                    for v in u:
                        videces = np.where(location_c == v)
                        r = revenue_c[videces]
                        rmean = r.mean()

                        if hidx + 1 == hours_nbr:
                            drmean = 0
                        else:
                            dr = distributed_revenue_vh[videces, hidx + 1]
                            drmean = dr.mean()

                        distributed_revenue_vh[videces, hidx] = rmean + drmean
                result_rsv[ridx, sidx, :] = distributed_revenue_vh[:, 0]

        result_rv = result_rsv.mean(axis=1)
        return result_rv


def _simulate_day(si: SimulateIn, trace=False) -> SimulateOut:
    """
    Simulate one day of vehicle movement.

    :param si: Initial location of vehicles. Array contains cell indexes. First array index is a row idx (row of vehicles), second array index is a vehicle idx.
    :type si: numpy.ndarray, shape=(R, V)

    :param trace: If true simulator records intermediate simulation states
    :type trace: bool

    :return: simulation result
    :rtype: SimulateOut
    """

    rows_nbr = si.location_row_vehicle.shape[0]
    vehicles_nbr = si.location_row_vehicle.shape[1]
    cells_nbr = si.region_cell.shape[0]
    hours_nbr = si.destination_hour_cell_sample.shape[0]
    regions_nbr = si.region_cell.max() + 1

    destination_sample_nbr = si.destination_hour_cell_sample.shape[2]
    r = np.random.randint(destination_sample_nbr)
    destination_sidx_hour_cell = np.zeros((hours_nbr, cells_nbr), dtype=np.int32)
    destination_sidx_hour_cell.fill(r)

    demand_sample_nbr = si.demand_sample_hour_cell.shape[0]

    out = SimulateOut()
    out.location_row_vehicle = si.location_row_vehicle
    out.revenue_row_simulation = np.zeros((rows_nbr, si.simulations_nbr), dtype=np.float32)
    out.rents_row_simulation = np.zeros((rows_nbr, si.simulations_nbr), dtype=np.uint16)
    out.satisfied_demand_row_simulation_region = np.zeros((rows_nbr, si.simulations_nbr, regions_nbr), dtype=np.float32)

    if trace:
        shape = (rows_nbr, si.simulations_nbr, vehicles_nbr, hours_nbr)
        out.revenue_row_simulation_vehicle_hour = np.zeros(shape, dtype=np.float32)
        out.location_row_simulation_vehicle_hour = np.zeros(shape, dtype=np.uint16)
        out.distance_row_simulation_vehicle_hour = np.zeros(shape, dtype=np.uint16)
        out.time_row_simulation_vehicle_hour = np.zeros(shape, dtype=np.uint16)
        out.rents_row_simulation_vehicle_hour = np.zeros(shape, dtype=np.uint8)

    assert_ndarray(si.region_cell, (cells_nbr,), np.uint16, "region_cell")
    assert_ndarray(
        si.destination_hour_cell_sample, (hours_nbr, cells_nbr, 10_000), np.uint16, "destination_hour_cell_sample"
    )
    assert_ndarray(si.demand_sample_hour_cell, (1000, hours_nbr, cells_nbr), np.uint16, "demand_sample_hour_cell")
    assert_ndarray(si.distance_hour_cell_cell, (hours_nbr, cells_nbr, cells_nbr), np.uint16, "distance_hour_cell_cell")
    assert_ndarray(si.time_hour_cell_cell, (hours_nbr, cells_nbr, cells_nbr), np.uint16, "time_hour_cell_cell")
    print("starting simulator")
    fast_simulate_day(
        si.location_row_vehicle,
        si.simulations_nbr,
        si.demand_sample_hour_cell,
        demand_sample_nbr,
        si.distance_hour_cell_cell,
        si.time_hour_cell_cell,
        si.destination_hour_cell_sample,
        destination_sidx_hour_cell,
        destination_sample_nbr,
        si.region_cell,
        regions_nbr,
        rows_nbr,
        cells_nbr,
        vehicles_nbr,
        hours_nbr,
        out.revenue_row_simulation,  # output param
        out.rents_row_simulation,  # output param
        out.satisfied_demand_row_simulation_region,  # output param
        trace,
        SEED,
        out.revenue_row_simulation_vehicle_hour,  # output param
        out.location_row_simulation_vehicle_hour,  # output param
        out.distance_row_simulation_vehicle_hour,  # output param
        out.time_row_simulation_vehicle_hour,  # output param
        out.rents_row_simulation_vehicle_hour,
    )  # output param

    return out
