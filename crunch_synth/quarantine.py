from typing import Dict, List, Tuple, Optional, Any
from bisect import bisect, bisect_left, bisect_right, insort

QuarantineEntry = Tuple[int, Any, int]  # Defines the type for entries in the quarantine


class Quarantine:
    """
    Handles quarantining of data points before they are eligible for processing.
    Stores tuples in the form (release_time, value, step), sorted by release_time.
    """

    def __init__(self):
        self.quarantine: List[QuarantineEntry] = []  # Stores tuples (release_time, value, step)

    def add(self, time: int, value, horizon: int, steps: list[int]) -> None:
        """
        Adds a new value to the quarantine list.
        The value will become available for prediction processing at `time + horizon`.

        Parameters
        ----------
        time : int
            The current time step.
        value : any
            The value to quarantine.
        horizon : int
            The number of time before the value is released.
        steps : list[int]
            Additional step-related metadata.
        """
        release_time = time + horizon
        # Insert while maintaining sorted order for `release_time`
        insort(self.quarantine, (release_time, value, steps))

    def pop(self, current_time: int) -> List[QuarantineEntry]:
        """
        Returns and removes **all** entries whose release_time <= current_time.

        This allows batch processing of all data points that have
        become available since the last check.

        Parameters
        ----------
        current_time : int
            The current logical time.

        Returns
        -------
        List[QuarantineEntry]
            A list of (release_time, value, step) tuples that are now eligible.
            Returns an empty list if no items are eligible.
        """

        # Find how many items are eligible
        idx = bisect_right(self.quarantine, current_time, key=lambda x: x[0])

        if idx <= 0:
            return []

        # Extract eligible entries
        eligible = self.quarantine[:idx]

        # Remove them from the quarantine
        self.quarantine = self.quarantine[idx:]

        return eligible


class QuarantineGroup:
    """
    Manages multiple Quarantine objects, one per asset.
    """

    def __init__(self):
        self.groups: Dict[str, Quarantine] = {}

    def _get_or_create(self, asset: str) -> Quarantine:
        """
        Returns the Quarantine object for `asset`,
        creating it if the asset does not yet exist.
        """
        if asset not in self.groups:
            self.groups[asset] = Quarantine()
        return self.groups[asset]

    def add(self, asset: str, time: int, value, horizon: int, steps: list[int]) -> None:
        """
        Adds a value to the quarantine for the given asset.
        """
        q = self._get_or_create(asset)
        q.add(time, value, horizon, steps)

    def pop(self, asset: str, current_time: int) -> List[QuarantineEntry]:
        """
        Pops all ready entries for a specific asset.

        Returns an empty list if:
            - asset does not exist
            - no entries are ready
        """
        if asset not in self.groups:
            return []

        return self.groups[asset].pop(current_time)

    def pop_all(self, current_time: int) -> Dict[str, List[QuarantineEntry]]:
        """
        Pops ready entries for **all assets** at once.

        Returns a dictionary:
            { asset: [entries...], ... }
        Only returns assets that have at least one ready entry.
        """
        result = {}

        for asset, quarantine in self.groups.items():
            ready = quarantine.pop(current_time)
            if ready:
                result[asset] = ready

        return result

    def assets(self) -> List[str]:
        """
        Returns the list of all assets that have quarantine data.
        """
        return list(self.groups.keys())
