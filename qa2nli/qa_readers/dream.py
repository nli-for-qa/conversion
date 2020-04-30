"""
[
    [
        [
            "M: How long have you been teaching in this middle school?",
            "W: For ten years. To be frank, I'm tired of teaching the same textbook for so long though I do enjoy being a teacher. I'm considering trying something new."
        ],
        [
            {
            "question": "What's the woman probably going to do?",
            "choice": [
            "To teach a different textbook.",
            "To change her job.",
            "To learn a different textbook."
            ],
            "answer": "To change her job."
            },

            {
            "question": "If the man and his wife go on the recommended package tour, how much should they pay?",
            "choice": [
            "$1,088.",
            "$1,958.",
            "$2,176."
            ],
            "answer": "$1,958."
            }
        ],
        "14-349"
    ],
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
logger = logging.getLogger(__name__)


class DreamReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'DreamJSON',
                 output_type: str = 'SingleQuestionSample'):

        if input_type != 'DreamJSON':
            raise ValueError(f"{input_type} unsupported")
        self.input_type = input_type
        self.output_type = output_type
        self.fitb_pattern = re.compile(r'_+')

    def _read_data(self, path: Path) -> Dict:
        with open(path) as f:
            samples = json.load(f)

        return samples

    def read(self, path: Path,
             return_dict: bool = False) -> List[Union[Sample, Dict]]:
        def reader_func(p: Path) -> List[Sample]:
            samples = self._read_data(p)
            # Give names to fields
            json_samples = []

            for s in samples:
                json_samples.append({
                    'passage': s[0],
                    'questions': s[1],
                    'id': s[2]
                })

            return json_samples

        if self.output_type == 'SingleQuestionSample':

            def sample_converter(x: Dict) -> Dict:
                # Do some preprocessing here
                # combine the dialogue sentences
                x['passage'] = ' '.join(x['passage'])
                # fix fitb format

                for q_n, q in enumerate(x['questions']):
                    x['questions'][q_n]['question'] = self.fitb_pattern.sub(
                        '_', x['questions'][q_n]['question'])

                # number the answer

                for q_n, question in enumerate(x['questions']):
                    # this will throw if answer does not match one of the
                    # choices exactly
                    idx = question['choice'].index(question['answer'])
                    question['answer'] = idx

                return x  # do nothing

            def aggregate_converter(
                    x: List[Dict]) -> List[SingleQuestionSample]:
                all_res = []

                for s in x:
                    para = s['passage']

                    for q_n, q in enumerate(s['questions']):
                        all_res.append(
                            SingleQuestionSample(
                                id=s['id'] + f"_{q_n}",
                                question=q['question'],
                                article=para,
                                options=q['choice'],
                                answer=q['answer']))

                return all_res

        else:
            raise ValueError(f"outpu_type {self.output_type} not supported")

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [s.__dict__ for s in output_samples]
        else:
            return output_samples
