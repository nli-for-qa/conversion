"""Convert the dataset into format understood by label studio"""

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
        logger.info("Read {} files from {}".format(i, folder.absolute()))

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


def split_id(idx: str):
    idx, ext = idx.split('.')
    head = idx.rstrip('0123456789')
    tail = idx[len(head):]

    return head, tail, ext


def write_in_race_format(data: List[Dict], dir_: Path):
    (dir_ / 'middle').mkdir(exist_ok=True)
    (dir_ / 'high').mkdir(exist_ok=True)
    high = 0
    middle = 0

    for example in data:
        idx = example['id']
        type_, n, ext = split_id(idx)

        if type_ == 'high':
            high += 1
        elif type_ == 'middle':
            middle += 1
        else:
            raise ValueError

        with open((dir_ / type_ / n).with_suffix('.' + ext), 'w') as f:
            json.dump(example, f)
    print(f'Written {middle} as middle and {high} as high')


@dataclass
class OriginalRaceSample(Sample):
    id: str
    article: str
    questions: List[str]
    options: List[List[str]]
    answers: List[str]


class RaceReader(DatasetReader):
    def __init__(self,
                 input_type: str = 'OriginalRaceSample',
                 output_type: str = 'SingleQuestionSample'):
        self.input_type = input_type
        self.output_type = output_type

    def read(self, path: Union[Path, str], return_dict=False) -> List[Dict]:
        path = Path(path)

        if self.input_type == 'OriginalRaceSample':

            def reader_func(p):
                return [OriginalRaceSample(**x) for x in _read_data(p)]
        elif self.input_type == 'SingleQuestionSample':

            def reader_func(p: Path) -> List[Mapping]:
                with open(p) as f:
                    return [SingleQuestionSample(**x) for x in json.load(f)]

        if (self.output_type == 'SingleQuestionSample'
                and self.input_type == 'SingleQuestionSample'):

            def sample_converter(x: Sample) -> Sample:
                return x

            def aggregate_converter(x: List[Sample]) -> List[Sample]:
                return x
        elif (self.output_type == 'SingleQuestionSample'
              and self.input_type == 'OriginalRaceSample'):

            def sample_converter(x: Sample) -> Sample:
                return x

            def aggregate_converter(x: List[Sample]) -> List[Sample]:
                all_res = []

                for s in x:
                    result = []
                    article = s.article

                    for i, (q, opts, a) in enumerate(
                            zip(s.questions, s.options, s.answers)):
                        result.append(
                            SingleQuestionSample(
                                id='_'.join([s.id, str(i)]),
                                article=article,
                                question=q,
                                options=opts,
                                answer=ANS_LETTER_TO_NUM[a],
                            ))
                    all_res.append(result)

                return [single for group in all_res for single in group]

        elif (self.output_type == 'SingleQuestionSingleOptionSample'
              and self.input_type == 'OriginalRaceSample'):

            def sample_converter(x: Sample) -> Sample:
                return x

            def aggregate_converter(x: List[Sample]) -> List[Sample]:
                all_res = []

                for s in x:
                    article = s.article

                    for i, (q, opts, a) in enumerate(
                            zip(s.questions, s.options, s.answers)):
                        q_id = '_'.join([s.id, str(i)])

                        for opt_num, opt in enumerate(opts):
                            all_res.append(
                                SingleQuestionSingleOptionSample(
                                    id='_'.join([q_id, str(opt_num)]),
                                    article=article,
                                    question=q,
                                    option=opt,
                                    label=int(ANS_LETTER_TO_NUM[a] == opt_num),
                                ))

                return all_res
        else:
            raise ValueError(
                f"input_type {self.input_type} and output_type {self.output_type} are not supported together"
            )

            # round trip to data validity checking

        input_samples = [sample_converter(s) for s in reader_func(path)]
        output_samples = aggregate_converter(input_samples)

        if return_dict:
            return [asdict(s) for s in output_samples]
        else:
            return output_samples
