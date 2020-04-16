from typing import Union, List, Tuple, Dict, Optional
from pathlib import Path
from .types import (Sample, SingleQuestionSample,
                    SingleQuestionSingleOptionSample, NLIWithOptionsSample,
                    PureNLISample)
import json
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)


class DatasetWriter(ABC):
    @abstractmethod
    def write(self, path: Path) -> None:
        pass


class JSONWriter(DatasetWriter):
    def __init__(self, data: List[Sample]):
        self.data = data

    def write(self, path: Path) -> None:
        output = [s.__dict__ for s in self.data]
        # call __dict__ to avoid copy
        with open(path, 'w') as f:
            json.dump(output, f)
        logger.info(f"Wrote to {path}")
