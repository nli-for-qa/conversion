"""
Original multirc format:
    {
        data: [
            {
                id: str,
                paragraph: {
                    text: {
                    },
                    questions: [
                        {
                            question: str,
                            sentences_used: [ int, ],
                            idx: int,
                            multisent: bool // don't know what this is
                            answers: [
                                {
                                    text: str,
                                    isAnswer: bool,
                                    scores: {} //empty
                                },
                                ...

                            ]
                        },
                        ...
                    ]
                }
            },
            ...
        ]
    }
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


@dataclass
class OriginalMultircSample(Sample):
    paragraph: Dict


html_tags = re.compile(r'<[^>]+>')
setence_tags = re.compile(r'Sent\s+\d+:')
html_plus_sentence_tags = re.compile(r"<[^>]+>|Sent\s+\d+:")


class MultircReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'OriginalMultircSample',
                 output_type: str = 'SingleQuestionSingleOptionSample'):
        self.input_type = input_type
        self.output_type = output_type

    def _read_data(self, path: Path) -> Dict:
        with open(path) as f:
            samples = json.load(f)

        return samples

    def read(self, path: Path,
             return_dict: bool = False) -> List[Union[Sample, Dict]]:

        if self.input_type == 'OriginalMultircSample':

            def reader_func(p: Path) -> List[Sample]:
                samples = ((self._read_data(p))['data'])
                # remove html

                for s in samples:
                    s['paragraph']['text'] = html_plus_sentence_tags.sub(
                        '', s['paragraph']['text'])

                return [OriginalMultircSample(**x) for x in samples]
        else:
            raise ValueError(f"input_type {self.input_type} not supported")

        if self.output_type == 'SingleQuestionSingleOptionSample':

            def sample_converter(
                    x: OriginalMultircSample) -> OriginalMultircSample:

                return x  # do nothing

            def aggregate_converter(
                    x: List[OriginalMultircSample]
            ) -> List[SingleQuestionSingleOptionSample]:
                all_res = []

                for s in x:
                    para = s.paragraph['text']

                    for q in s.paragraph['questions']:
                        for ans_i, a in enumerate(q['answers']):

                            all_res.append(
                                SingleQuestionSingleOptionSample(
                                    id=s.id + f"_{q['idx']}" + f"_{ans_i}",
                                    article=para,
                                    question=q['question'],
                                    option=a['text'],
                                    label=int(a['isAnswer'])))

                return all_res

        else:
            raise ValueError(f"outpu_type {self.output_type} not supported")

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [s.__dict__ for s in output_samples]
        else:
            return output_samples
