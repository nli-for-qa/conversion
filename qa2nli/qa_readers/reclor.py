"""
Original Format : JSON

[
    {
        id_string: str
        context: str
        question: str
        answers: List[str] (length 4)
        label: int (0 to 3)
    },
    ...
]
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Generator, Optional, Union
from copy import deepcopy
import itertools
import re
import logging
from .reader import DatasetReader
from .types import (Sample, SingleQuestionSample,
                    SingleQuestionSingleOptionSample, NLIWithOptionsSample,
                    PureNLISample)

from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)


class ReclorReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'DreamJSON',
                 output_type: str = 'SingleQuestionSample'):

        if input_type != 'DreamJSON':
            raise ValueError(f"{input_type} unsupported")
        self.input_type = input_type
        self.output_type = output_type

    def _read_data(self, path: Path) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)

        return data

    def read(self, path: Path,
             return_dict: bool = False) -> List[Union[Sample, Dict]]:
        def reader_func(p: Path) -> List[Sample]:
            samples = self._read_data(p)
            # Convert to single question type here only

            return [
                SingleQuestionSample(
                    id=s['id_string'],
                    question=s['question'],
                    article=s['context'],
                    options=s['answers'],
                    answer=s['label']) for s in samples
            ]

        if self.output_type == 'SingleQuestionSample':

            def sample_converter(x: Dict) -> Dict:
                return x  # do nothing

            def aggregate_converter(
                    x: List[Dict]) -> List[SingleQuestionSample]:

                return x

        else:
            raise ValueError(f"outpu_type {self.output_type} not supported")

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [s.__dict__ for s in output_samples]
        else:
            return output_samples
