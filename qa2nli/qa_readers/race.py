"""Convert the dataset into format understood by label studio"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Generator
from copy import deepcopy
import itertools
import logging
logger = logging.getLogger(__name__)

ANS_LETTER_TO_NUM = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
ANS_NUM_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def read_data(p: Path, qa_only: bool = True) -> List[Dict]:
    samples = _read_data(p)

    qa_samples = []

    for sample in samples:
        for q_num, (q, opts, a) in enumerate(
                zip(sample['questions'], sample['options'],
                    sample['answers'])):
            qa_sample = {'id': sample['id'] + '_' + str(q_num)}
            qa_sample['question'] = q
            qa_sample['answer'] = opts[ANS_LETTER_TO_NUM[a]]
            qa_samples.append(qa_sample)

            if not qa_only:
                qa_sample['article'] = sample['article']
                qa_sample['options'] = opts
                qa_sample['answer_opt'] = a

    return qa_samples


def process_samples_lazy(qa_samples: List[Dict]) -> Generator:

    for qa_sample in qa_samples:

        for i, opt in enumerate(qa_sample['options']):
            sample = {'id': qa_sample['id'] + '_' + ANS_NUM_TO_LETTER[i]}
            sample['premise'] = qa_sample['article']
            sample['question'] = qa_sample['question']
            sample['option'] = opt
            sample['label'] = i == ANS_LETTER_TO_NUM[qa_sample['answer_opt']]
            yield sample


def _read_data(p: Path) -> List[Mapping]:
    samples = []
    suffix = ['high', 'middle']

    for s in suffix:
        folder = p / s

        for i, file in enumerate(folder.glob('*.txt')):
            with open(file) as f:
                sample = json.load(f)
                samples.append(sample)
        print("Read {} files from {}".format(i, folder.absolute()))

    return samples
