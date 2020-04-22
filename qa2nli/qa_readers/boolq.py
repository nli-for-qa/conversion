"""
Original BoolQ format (dev.jsonl):
    {
        title: str,
        passage: str,
        question: str,
        answer: bool

    }\n
    {
        title: str,
        passage: str,
        question: str,
        answer: bool

    }\n
    ...
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
import jsonlines as jsonl
logger = logging.getLogger(__name__)


@dataclass
class OriginalBoolQSample(Sample):
    title: str
    passage: str
    question: str
    answer: bool


class BoolQReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'OriginalBoolQSample',
                 output_type: str = 'SingleQuestionSingleOptionSample'):
        self.input_type = input_type
        self.output_type = output_type

    def _read_data(self, path: Path) -> List[Dict]:
        with jsonl.open(path) as reader:
            samples = list(reader)

        return samples

    def read(self, path: Path,
             return_dict: bool = False) -> List[Union[Sample, Dict]]:

        if self.input_type != 'OriginalBoolQSample' and self.output_type != 'SingleQuestionSingleOptionSample':
            raise ValueError(f"outpu_type {self.output_type} not supported")

        def reader_func(p: Path) -> List[OriginalBoolQSample]:

            return [
                OriginalBoolQSample(id=s['title'], **s)

                for s in self._read_data(p)
            ]

        def sample_converter(x: OriginalBoolQSample) -> PureNLISample:
            return SingleQuestionSingleOptionSample(
                id=x.id,
                article=x.passage,
                question=x.question +
                ' ?',  # dataset does not have question mark
                option='yes',
                label=int(x.answer))

        def aggregate_converter(x: List[SingleQuestionSingleOptionSample]
                                ) -> List[SingleQuestionSingleOptionSample]:

            return x

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [s.__dict__ for s in output_samples]
        else:
            return output_samples
