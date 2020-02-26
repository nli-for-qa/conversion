"""Convert the dataset into format understood by label studio"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Generator, Optional
from copy import deepcopy
import itertools
import re
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


def read_nli_data(p: Path) -> List[Dict]:
    """Read dataset which has been converted to nli form"""
    with open(p) as f:
        data = json.load(f)

    return data


def get_qa_filename_from_nli_sample(nli_sample: Dict) -> str:
    filename, question_number, option = (nli_sample['id']).split('_')

    return filename


def get_matching_qa_sample(nli_sample: Dict,
                           qa_data_dir: Path) -> Optional[Dict]:
    filename, question_number, option = (nli_sample['id']).split('_')
    qa_id = ('_').join([filename, question_number])

    return get_qa_sample(qa_id, qa_data_dir)


def get_qa_sample(qa_id: str, qa_data_dir: Path) -> Dict:
    filename, question_number = (qa_id).split('_')
    m = re.match(r"(high|middle)(\d+\.txt)", filename)

    if not m:
        raise ValueError("id {} is invalid".format(qa_id))
    filepath = qa_data_dir / m.group(1) / m.group(2)

    with open(filepath) as f:
        qa_sample = json.load(f)
        res = {
            'id': qa_id,
            'article': qa_sample['article'],
            'question': qa_sample['questions'][int(question_number)],
            'answer': qa_sample['answers'][int(question_number)],
            'options': qa_sample['options'][int(question_number)]
        }

    return res


def conversion_successful(nli_sample: Dict) -> bool:
    return nli_sample['conversion_success']
