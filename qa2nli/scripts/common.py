from multiprocessing import Pool, current_process, Queue
from pathlib import Path
from typing import Any, List, Dict
from qa2nli.converters.processors import Preprocessor, Postprocessor
import multiprocessing
import itertools
import argparse
from qa2nli.qa_readers.reader import SingleQuestionSample, Sample
import json
import logging
logger = logging.getLogger(__name__)

model = None


def init(device: int, model_type: str, model_path: Path, model_class: Any,
         preprocessor_args: Dict, postprocessor_args: Dict) -> None:
    """ Initialize the model"""
    global model

    if model_type == 'dummy':

        def converter(qs: List[str], os: List[str]) -> List[str]:

            return [q + ' ' + o for q, o in zip(qs, os)]

        model = converter
    else:
        model = model_class(
            model_path,
            device=device,
            preprocessor=Preprocessor(),
            postprocessor=Postprocessor(**postprocessor_args))
    SingleQuestionSample.to_nli_converter = model
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


def run_inference(input_sample: Sample,
                  output_dir: Path = None,
                  target_type: str = 'NLIWithOptionsSample',
                  overwrite: bool = False) -> None:
    """ Called in each process separately.
    Make sure the output_dir exists"""
    idx = input_sample.id
    # check if exists
    file_ = output_dir / idx

    if ((file_).is_file()) and (not overwrite):
        logger.debug(f"Skipping {idx} as {file_} exists")

        return None  # nothing to do

    # use the global model object
    converted: List = input_sample.to(target_type)

    for ex in converted:
        with open(file_, 'w') as f:
            json.dump(ex.__dict__, f)
            logger.debug(
                f"process {multiprocessing.current_process().pid} wrote {idx}")

    return


def generate_infer_entry(output_dir, target_type, overwrite):
    return run_inference()


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('convert')
    parser.add_argument(
        '--model_type', default='dummy', choices=['dummy', 'bart'])
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
        '--input_reader', default='race_reader', choices=['race_reader'])
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