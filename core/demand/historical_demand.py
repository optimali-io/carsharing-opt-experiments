import logging
from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
log = logging.getLogger("core")


def convert_geometry_to_coords(geometry_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Fake function to convert WKT from backend representation into lon, lat"""
    lons: pd.Series = geometry_series.apply(lambda s: float(s.split("POINT ")[1].split(" ")[0].replace("(", "")))
    lats: pd.Series = geometry_series.apply(lambda s: float(s.split("POINT ")[1].split(" ")[1].replace(")", "")))
    return lons, lats


class GetHistoricalDemand:
    """Get demand values computed based on events and rents"""

    time_before_rent = pd.Timedelta(seconds=1200)  # maximum time between reservation and start of rent in seconds
    time_between_demands = pd.Timedelta(seconds=3600)  # time in seconds

    cell_size = 530
    distance_between_demands = (2 * cell_size) ** 2
    transformer = Transformer.from_proj("epsg:4326", "epsg:2180", always_xy=True)

    timeseries_columns = ["lon", "lat", "cell_id", "time", "device_id", "info"]
    start_rent_columns = ["start_lon", "start_lat", "start_cell_id", "start_date", "device_id", "start_rent"]
    end_rent_columns = ["end_lon", "end_lat", "end_cell_id", "end_date", "device_id", "end_rent"]
    appstart_columns = ["lon", "lat", "cell_id", "event_datetime", "deviceid"]

    def __init__(self, rents_df: pd.DataFrame, appstarts_df: pd.DataFrame):
        self.rents_df: pd.DataFrame = rents_df
        self.appstarts_df: pd.DataFrame = pd.DataFrame(appstarts_df[self.appstart_columns]) if appstarts_df is not None else pd.DataFrame(columns=self.appstart_columns)
        self.appstarts_df["info"] = "appstart"
        self.appstarts_df.columns = self.timeseries_columns

    def _create_events_timeseries(self) -> pd.DataFrame:
        """
        Method converts rents dataframe into rent events timeseries by splitting every rent into
        "reservation", "start", "stop", joining with appstarts and sorting events by time.
        """

        self.rents_df["start_rent"] = "start_rent"
        self.rents_df["end_rent"] = "end_rent"

        log.info("converting geometries to coords")

        reservations_df = pd.DataFrame(self.rents_df[self.start_rent_columns])
        reservations_df.columns = self.timeseries_columns
        reservations_df["info"] = "reservation"
        reservations_df["time"] = reservations_df["time"] - self.time_before_rent

        start_df = pd.DataFrame(self.rents_df[self.start_rent_columns])
        start_df.columns = self.timeseries_columns

        end_df = pd.DataFrame(self.rents_df[self.end_rent_columns])
        end_df.columns = self.timeseries_columns

        log.info("concatenating frames into single timeseries frame")
        timeseries_df = (
            pd.concat([self.appstarts_df, reservations_df, start_df, end_df])
            .sort_values(by="time")
            .reset_index(drop=True)
        )

        return timeseries_df

    @staticmethod
    def _filter_appstarts_from_unfinished_rents(events_timeseries: pd.DataFrame) -> pd.DataFrame:
        """
        If first rent-related event in series is "end-rent" all appstarts before it should be removed.
        If last rent-related event in series is "start-rent" all appstarts after it should be removed.
        """
        start_rents_indices = events_timeseries[events_timeseries["info"] == "start_rent"].index
        end_rents_indices = events_timeseries[events_timeseries["info"] == "end_rent"].index
        if start_rents_indices.shape[0] > 0 and end_rents_indices.shape[0] > 0:
            # There is at least one finished rent.
            first_start_rent_index = start_rents_indices[0]
            first_end_rent_index = end_rents_indices[0]

            last_start_rent_index = start_rents_indices[-1]
            last_end_rent_index = end_rents_indices[-1]

            if first_end_rent_index < first_start_rent_index:
                events_timeseries.loc[: first_end_rent_index - 1, "info"] = "drop"
            if last_end_rent_index < last_start_rent_index:
                events_timeseries.loc[last_start_rent_index + 1 :, "info"] = "drop"
            events_timeseries = events_timeseries.loc[events_timeseries["info"] != "drop"]
            return events_timeseries.reset_index(drop=True)
        elif start_rents_indices.shape[0] > 0:
            # The only rent is started and unfinished.
            # Everything after last start_rent should be dropped because rent hasn't finished yet.
            first_start_rent_index = start_rents_indices[0]
            return events_timeseries.iloc[: first_start_rent_index + 1, :].reset_index(drop=True)
        elif end_rents_indices.shape[0] > 0:
            # The only rent is rent that started before timeseries started (very long rent).
            # Everything before first end_rent should be dropped because it happened during the rent.
            first_end_rent_index = end_rents_indices[0]
            return events_timeseries.iloc[first_end_rent_index:, :].reset_index(drop=True)

        return events_timeseries

    def _are_events_close(self, previous_row: pd.Series, current_row: pd.Series) -> bool:
        """Check if events should be considered separate."""
        dist = (current_row.TRLon - previous_row.TRLon) ** 2 + (current_row.TRLat - previous_row.TRLat) ** 2
        return (
            current_row.time - previous_row.time <= self.time_between_demands and dist <= self.distance_between_demands
        )

    def _find_demand(self, db: pd.DataFrame) -> pd.DataFrame:
        """Removes all events from user specific timeseries that are not considered demand."""
        drop_list = []
        rent = 0
        next_event_sd = False

        db = self._filter_appstarts_from_unfinished_rents(db)

        db["TRLon"], db["TRLat"] = self.transformer.transform(list(db["lon"]), list(db["lat"]))

        # create two helper rows
        db_row = namedtuple("Pandas", "Index lon lat cell_id time device_id info TRLon TRLat")
        prev_sd = db_row(
            -1,
            db.iat[0, 0],
            db.iat[0, 1],
            0,
            pd.Timestamp("1900-01-01T12").tz_localize("utc"),
            "None",
            "appstart",
            db.iat[0, 6],
            db.iat[0, 7],
        )
        prev_ev = db_row(
            -1,
            db.iat[0, 0],
            db.iat[0, 1],
            0,
            pd.Timestamp("1900-01-01T12").tz_localize("utc"),
            "None",
            "appstart",
            db.iat[0, 6],
            db.iat[0, 7],
        )

        for row in db.itertuples():
            if row.info == "start_rent":
                if next_event_sd:
                    # this event should be start demand
                    if self._are_events_close(prev_ev, row):
                        # drop last start demand if this is the same series
                        drop_list.append(prev_sd.Index)
                    prev_sd = row
                    next_event_sd = False
                else:
                    # start demand for this start rent has been already selected
                    drop_list.append(row.Index)
                prev_ev = row
            elif row.info == "end_rent":
                # just drop it and decrease rent number
                rent -= 1
                drop_list.append(row.Index)
                prev_ev = row
            elif row.info == "appstart":
                if next_event_sd:
                    # this event should be start demand
                    if self._are_events_close(prev_ev, row):
                        # drop last start demand if this is the same series
                        drop_list.append(prev_sd.Index)
                    prev_sd = row
                    next_event_sd = False
                else:
                    if self._are_events_close(prev_ev, row) or rent > 0:
                        # the same series or in the middle of rent
                        drop_list.append(row.Index)
                    else:
                        # start demand should be here
                        prev_sd = row
                prev_ev = row
            elif row.info == "reservation":
                if rent == 0:
                    # next event or start rent should be start demand
                    next_event_sd = True
                # just drop it and increase rent number
                rent += 1
                drop_list.append(row.Index)
            else:
                raise ValueError(f"Unknown type of Info in user events: {row.info}")

        dbf = db.drop(drop_list)
        return dbf

    def get_demand(self) -> pd.DataFrame:
        """
        Method creates pandas dataframe with timseries of events of type [appstart, start_rent].
        Dataframe columns:
        - lon: float
        - lat: float
        - cell_id: int or str
        - time: datetime
        - device_id: str
        """
        log.info("creating events timeseries")
        events_timeseries: pd.DataFrame = self._create_events_timeseries()
        log.info("grouping frames by user")
        events_timeseries_by_user = [
            group[1].reset_index(drop=True) for group in events_timeseries.groupby("device_id")
        ]
        log.info("finding demand per user")
        filtered_events_per_user = [self._find_demand(dframe) for dframe in events_timeseries_by_user]
        log.info("concatenating demand frames")
        demand = pd.concat(filtered_events_per_user).sort_values(by="time").reset_index(drop=True)
        return demand

    @staticmethod
    def aggregate_demand(demand: pd.DataFrame, n_cells: int) -> np.ndarray:
        """
        Method aggregates demand points into numpy array of given 2d shape (n_hours, n_cells).

        :param demand - precalculated pandas DataFrame with demand points
        :param n_cells - number of cells in zone
        """
        demand["hour"] = demand["time"].apply(lambda d: d.hour)

        demand_hour_cell: np.array = np.zeros((24, n_cells))
        for row in demand.itertuples():
            demand_hour_cell[row.hour, int(row.cell_id)] += 1
        return demand_hour_cell
