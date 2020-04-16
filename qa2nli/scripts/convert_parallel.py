from typing import List
import argparse
import logging
from pathlib import Path
from qa2nli.scripts.common import (get_default_parser,
                                   create_shared_devices_queue, run_inference,
                                   consume_device_queue_and_init)
from qa2nli.converters.bart.predictor import BartConverter
from qa2nli.converters.processors import Preprocessor, Postprocessor
from qa2nli.qa_readers.race import RaceReader
from qa2nli.qa_readers.reader import SingleQuestionSample, Sample
from tqdm import tqdm
import multiprocessing
from functools import partial
logger = logging.getLogger(__file__)
LEVEL = logging.INFO

logging.basicConfig(
    format="%(process)d - %(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=LEVEL)


def get_parser() -> argparse.ArgumentParser:
    parser = get_default_parser()
    parser.add_argument(
        '--num_processes',
        type=int,
        default=-1,
        help='Number of inference processes to be run.'
        ' If on cpu, use only 1 as you would run out of RAM.'
        'Setting to -1 (default) will set it as len(devices)')
    parser.add_argument(
        '--per_process_chunk_size',
        default=100,
        type=int,
        help='Chunk size for pool.imap()')

    return parser


def main(args: argparse.Namespace) -> None:

    # checks on args

    if args.num_processes == -1:
        args.num_processes = len(args.device)

    if args.debug:
        logger.info("Setting all loggers to DEBUG as --debug was passed")
        loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]

        for _ in loggers:
            _.setLevel(logging.DEBUG)
    # make sure output dir exists
    outfile = (args.output / args.set).with_suffix('.json')

    if outfile.is_file():
        if args.overwrite:
            pass
        else:
            raise RuntimeError(
                f"{ args.output / args.set} already exists. If want to overwrite set --overwrite flag"
            )
    else:
        # make sure the dir exists
        args.output.mkdir(parents=True, exist_ok=True)

    output_cache_dir = args.output / 'cache'
    output_cache_dir.mkdir(exist_ok=True)
    logger.info(f"Setting output cache to {output_cache_dir}")

    # Read input data
    inp_dir = args.input_data / args.set

    if args.input_reader == 'race_reader':
        inp_data = RaceReader().read(inp_dir)
    else:
        raise ValueError
    # check model class

    if args.model_type == 'bart':
        model_class = BartConverter
    else:
        model_class = None
    logger.info(f"Model class used is {model_class}")
    # setup device queue
    devices_assignment_queue = create_shared_devices_queue(
        args.device, args.num_processes)

    # create the inference func
    # need to create it before the pool is initizalied
    infer = partial(
        run_inference,
        output_dir=output_cache_dir,
        target_type=args.target_type,
        overwrite=(not args.resume_from_cache))

    # init process pool, giving one device assignment to each from the queue
    process_pool = multiprocessing.Pool(
        processes=args.num_processes,
        initializer=consume_device_queue_and_init,
        initargs=(
            devices_assignment_queue,
            args.model_type,
            args.model_path,
            model_class,
            dict(),  # preprocessor_args
            dict(
                cleaner=args.postprocess_cleaner,
                sentence_splitter=args.
                postprocess_splitter)  # postprocessor_args
        ))
    # do the inference
    jobs = process_pool.imap_unordered(
        infer, inp_data, chunksize=args.per_process_chunk_size)

    for job in jobs:
        pass  # jobs is an iter so need to iterate

    # recombine cache to


if __name__ == '__main__':
    main(get_parser().parse_args())