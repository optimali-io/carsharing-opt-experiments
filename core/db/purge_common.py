import datetime
import logging
import re
from typing import Dict, List

log = logging.getLogger(__name__)


class PurgeEntry:
    """Class represents purge entry."""

    def __init__(self):
        self.key: str = None
        self.date: datetime.date = None
        self.type: str = None
        self.remove: bool = None

    def __repr__(self):
        return f"{self.date} {self.type} {self.remove} {self.key}"


def select_to_remove(
    purge_entries: List[PurgeEntry], today: datetime.date, remove_rules: Dict[str, int]
) -> List[PurgeEntry]:
    """
    Select PurgeEntry to remove from candidate list.

    :param purge_entries: List of purge candidates
    :type purge_entries: List[PurgeEntry]

    :param today: Today
    :type today: datetime.date

    :param remove_rules: Rules describing how many days keep files of specific type
    :type remove_rules: Dict[str, int]

    :return: List of PurgeEntries to remove
    :rtype: List[PurgeEntry]
    """
    _recognize_date_and_type(purge_entries)
    _mark_remove(purge_entries, today, remove_rules)
    purge_entries = _filter_remove(purge_entries)
    return purge_entries


def _recognize_date_and_type(purge_entries: List[PurgeEntry]):
    """
    Set date and type for files to be removed.
    :param purge_entries: List of PurgeEntry objects
    """
    file_date_file_type_pattern = r".*zone/[0-9]+/date/([0-9]{4})/([0-9]{2})/([0-9]{2})/([_a-z]+).npy"
    for pe in purge_entries:
        m = re.match(file_date_file_type_pattern, pe.key)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3))
            pe.date = datetime.date(year, month, day)
            pe.type = m.group(4)


def _mark_remove(purge_entries: List[PurgeEntry], today: datetime.date, remove_rules: Dict[str, int]):
    """
    Mark files to be removed if they fulfill provided conditions.
    :param purge_entries: List of PurgeEntry objects
    :param today: dt.date
    :param remove_rules: dict containing rules for files to be removed
    """
    for pe in purge_entries:
        if pe.type in remove_rules:
            remove_after_days = remove_rules[pe.type]
            entry_age = today - pe.date
            if entry_age.days > remove_after_days:
                pe.remove = True
            else:
                pe.remove = False


def _filter_remove(purge_entries: List[PurgeEntry]) -> List[PurgeEntry]:
    """
    Return only files to be removed.
    :param purge_entries: List of PurgeEntry objects
    :return: List of PurgeEntry objects
    """
    return [e for e in purge_entries if e.remove]


def log_list(purge_entries: List[PurgeEntry]):
    """
    Log list of PurgeEntry objects.

    :param purge_entries: purge entries
    :type purge_entries: List[PurgeEntry]
    """
    for pe in purge_entries:
        log.debug(pe)
