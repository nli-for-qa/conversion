from typing import List
import argparse
import logging
from pathlib import Path
from qa2nli.converters.bart.predictor import BartConverter
from qa2nli.converters.processors import Preprocessor, Postprocessor
from qa2nli.qa_readers.race import RaceReader
from qa2nli.qa_readers.reader import SingleQuestionSample, Sample
from qa2nli.qa_readers.writer import JSONWriter
from tqdm import tqdm
logger = logging.getLogger(__file__)
LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('convert_race')
    parser.add_argument(
        '--model_path',
        required=True,
        type=Path,
        help='Path to BART model checkpoint')
    parser.add_argument(
        '--device', type=int, default=-1, help='-1 for cpu 0 for gpu')
    parser.add_argument(
        '--input_data', type=Path, help='Path to input data directory')
    parser.add_argument('--set', default='dev', help='dev, train or test')
    parser.add_argument('--output', type=Path, help='Output dir path')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='to overwrite existing output file or not')
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

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace) -> None:
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
    inp_dir = args.input_data / args.set
    race_data = RaceReader().read(inp_dir)
    converter = BartConverter(
        args.model_path,
        device_number=args.device,
        preprocessor=Preprocessor(),
        postprocessor=Postprocessor(
            cleaner=args.cleaner, sentence_splitter=args.splitter))
    SingleQuestionSample.to_nli_converter = converter

    converted: List[Sample] = []

    for ex in tqdm(race_data):
        converted.extend(ex.to(args.target_type))
    writer = JSONWriter(converted)
    logger.info(f'Writing to {outfile}')
    writer.write(outfile)


if __name__ == '__main__':
    main(parse_args())
