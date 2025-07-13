import datetime
import logging
import os
from pathlib import Path
from typing import List

from core.db.purge_common import PurgeEntry, log_list, select_to_remove

log = logging.getLogger(__name__)


class PurgeLocal:
    """Class represents local purge process."""

    def __init__(self, today: datetime.date, s3_files_local_cache_dir: str) -> None:
        """
        Constructor

        :param today: Today
        :type today: datetime.date

        :param s3_files_local_cache_dir: Local cache directory for S3 files
        :type s3_files_local_cache_dir: str
        """
        self._today = today
        self._s3_files_local_cache_dir = s3_files_local_cache_dir
        self._remove_rules = {"demand_prediction": 10, "revenue_estimation": 10, "revenue_parameters": 10}
        self._purge_entries: List[PurgeEntry] = None

    def purge(self) -> None:
        """
        Execute local files purge
        """
        log.info("Purge local begin")
        self._list_files()
        self._purge_entries = select_to_remove(self._purge_entries, self._today, self._remove_rules)
        self._remove_files()
        log_list(self._purge_entries)
        log.info("Purge local done")

    def _list_files(self) -> None:
        """Create a list of all files in a cache (local filesystem)."""
        self._purge_entries = []
        for root, directories, files in os.walk(self._s3_files_local_cache_dir):
            for filename in files:
                absolute_path = os.path.join(root, filename)
                purge_entry = PurgeEntry()
                purge_entry.key = absolute_path
                self._purge_entries.append(purge_entry)

    def _remove_files(self) -> None:
        """Remove all files from a list."""
        for pe in self._purge_entries:
            file_path = Path(pe.key)
            file_path.unlink(missing_ok=True)
