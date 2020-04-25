from multiprocessing import Pool, current_process, Queue
from pathlib import Path
from typing import Any, List, Dict
from qa2nli.converters.processors import Preprocessor, Postprocessor, PostprocessorBase
from qa2nli.converters.base import Converter
from qa2nli.converters.rule.predictor import RuleBasedConverter
import multiprocessing
import itertools
import argparse
from qa2nli.qa_readers.reader import SingleQuestionSample, SingleQuestionSingleOptionSample, Sample
import json
import re
import logging
logger = logging.getLogger(__name__)

model = None


def init(device: int, model_type: str, model_path: Path, model_class: Any,
         preprocessor_args: Dict, postprocessor_args: Dict) -> None:
    """ Initialize the model"""
    global model
    # This is a hack

    if model_class in [Converter, RuleBasedConverter]:
        logger.warning(
            f"Ignoring post-processor args for {model_class.__class__.__name__}"
        )
        postprocessor = PostprocessorBase()
    else:
        postprocessor = Postprocessor(**postprocessor_args)
    model = model_class(
        model_path,
        device_number=device,
        preprocessor=Preprocessor(),
        postprocessor=postprocessor)
    SingleQuestionSample.to_nli_converter = model
    SingleQuestionSingleOptionSample.to_nli_converter = model
    logger.info(
        f"Initialized device {device} on pid {multiprocessing.current_process().pid}"
    )


def consume_device_queue_and_init(devices: multiprocessing.Queue,
                                  model_type: str, model_path: Path,
                                  model_class: Any, preprocessor_args: Dict,
                                  postprocessor_args: Dict) -> None:

    if not devices.empty():
        device = devices.get()
        init(device, model_type, model_path, model_class, preprocessor_args,
             postprocessor_args)
    else:
        raise RuntimeError


def create_shared_devices_queue(devices: List[int],
                                total_entries: int) -> multiprocessing.Queue:
    q: multiprocessing.Queue = multiprocessing.Queue()

    for entry_number, device_id in zip(
            range(total_entries), itertools.cycle(devices)):
        q.put(device_id)

    return q


_backslash_pattern = re.compile(r"\/")


def valid_filename(idx: str) -> str:
    return _backslash_pattern.sub("_", idx)


def run_inference(input_sample: Sample,
                  output_dir: Path = None,
                  target_type: str = 'NLIWithOptionsSample',
                  overwrite: bool = False) -> None:
    """ Called in each process separately.
    Make sure the output_dir exists"""
    idx = input_sample.id
    # check if exists
    file_ = output_dir / valid_filename(idx)
    logger.info(
        f"pid {multiprocessing.current_process().pid} processing id {idx}")

    if ((file_).is_file()) and (not overwrite):
        logger.debug(f"Skipping {idx} as {file_} exists")

        return None  # nothing to do

    # use the global model object
    converted: List = input_sample.to(target_type)
    logger.info(
        f"pid {multiprocessing.current_process().pid} converted id {idx}")

    for ex in converted:
        with open(file_, 'w') as f:
            json.dump(ex.__dict__, f)
            logger.debug(
                f"process {multiprocessing.current_process().pid} wrote {idx}")

    return


def read_cache_dir(cache_dir: Path) -> List[Dict]:
    logger.info(f"Reading cache from {cache_dir}")
    samples = []

    for potential_file in cache_dir.iterdir():
        if potential_file.is_file():
            with open(potential_file) as f:
                samples.append(json.load(f))

    return samples


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('convert')
    parser.add_argument(
        '--model_type',
        default='dummy',
        choices=['dummy', 'bart', 'just_question', 'const', 'concat', 'rule'])
    parser.add_argument(
        '--model_path',
        required=True,
        type=Path,
        help='Path to BART model checkpoint')
    parser.add_argument(
        '--device',
        nargs='+',
        type=int,
        default=-1,
        help='-1 for cpu 0 for gpu')

    parser.add_argument(
        '--input_reader',
        default='race_reader',
        choices=[
            'dream_reader', 'race_reader', 'multirc_reader', 'boolq_reader'
        ])
    parser.add_argument(
        '--input_data', type=Path, help='Path to input data directory')
    parser.add_argument('--set', default='dev', help='dev, train or test')
    parser.add_argument('--output', type=Path, help='Output dir path')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='to overwrite existing output file or not')
    parser.add_argument(
        '--resume_from_cache',
        action='store_true',
        help='set this to resume from a partially finished job')
    parser.add_argument(
        '--target_type',
        default='NLIWithOptionsSample',
        choices=['PureNLISample', 'NLIWithOptionsSample'])
    parser.add_argument(
        '--postprocess_cleaner', default=None, choices=[None, 'remove_dots'])
    parser.add_argument(
        '--postprocess_splitter',
        default='period',
        choices=['spacy', 'period'])
    parser.add_argument('--debug', action='store_true')

    return parser


def parse_args(parser: argparse.Namespace):
    return parser.parse_args()
