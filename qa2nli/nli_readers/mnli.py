"""
Original MNLI format (JSONLINES):
    {
        sentence1: str,
        sentence2: str,
        gold_label: Choice[contradiction, neutral, entailment],
        annotator_labels: List[str],
        genre: str
        pairID: str (unique)
        promptID: str,
        setence1_parse: str
        sentence1_binary_parse:str
        setence2_parse: str
        sentence2_binary_parse:str
    }

"""
import jsonlines as jsonl
from pathlib import Path
from ..qa_readers.reader import DatasetReader
from typing import Dict, List, Mapping, Generator, Optional, Union
from ..qa_readers.types import (Sample, SingleQuestionSample,
                                SingleQuestionSingleOptionSample,
                                NLIWithOptionsSample, PureNLISample)

import logging
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)


@dataclass
class OriginalMNLISample(Sample):
    sentence1: str
    sentence2: str
    gold_label: str
    annotator_labels: List[str]
    genre: str
    pairID: str
    promptID: str
    sentence1_parse: str
    sentence1_binary_parse: str
    sentence2_parse: str
    sentence2_binary_parse: str


class MNLIReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'OriginalMNLISample',
                 output_type: str = 'PureNLISample'):
        self.input_type = input_type
        self.output_type = output_type

    def _read_data(self, path: Path) -> List[Dict]:
        with jsonl.open(path) as f:
            samples = list(f)

        return samples

    def read(self, path: Path,
             return_dict: bool = False) -> List[Union[Sample, Dict]]:

        if self.input_type != 'OriginalMNLISample' or \
                self.output_type != 'PureNLISample':
            ValueError(
                f"input_type {self.input_type} and/or output_type {self.output_type} not supported"
            )

        def reader_func(p: Path) -> List[Sample]:
            return [
                OriginalMNLISample(id=s['pairID'], **s)

                for s in self._read_data(p)
            ]

        def sample_converter(x: OriginalMNLISample) -> PureNLISample:
            return PureNLISample(
                id=x.pairID,
                premise=x.sentence1,
                hypothesis=x.sentence2,
                label=int(x.gold_label == 'entailment'),
                meta=dict(dataset='mnli', genre=x.genre))

        def aggregate_converter(x: List[PureNLISample]) -> List[PureNLISample]:
            return x

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [s.__dict__ for s in output_samples]
        else:
            return output_samples
