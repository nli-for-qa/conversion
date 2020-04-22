import pytest
from typing import List
from qa2nli.qa_readers.boolq import BoolQReader
from qa2nli.qa_readers.types import SingleQuestionSingleOptionSample
from qa2nli.qa_readers.reader import SingleQuestionSample, ConversionManager, PureNLISample
from qa2nli.qa_readers.writer import JSONWriter
from pathlib import Path
from dataclasses import asdict
from .data import (boolq_data_path, boolq_data)
from .converters import bart_with_spacy_converter
import json
import itertools
import tqdm
import logging
logger = logging.getLogger(__file__)


def test_boolq_reader(boolq_data_path):
    reader = BoolQReader()
    data = reader.read(boolq_data_path)

    for i in range(3):
        logger.info(data[i])


def test_boolq_to_PureNLI_bart_with_Prepostprocessors(
        boolq_data_path, bart_with_spacy_converter):
    reader = BoolQReader()
    SingleQuestionSingleOptionSample.to_nli_converter = bart_with_spacy_converter
    data: List[SingleQuestionSingleOptionSample] = reader.read(boolq_data_path)

    converted = [sample.to('PureNLISample') for sample in data]

    for i in range(0, len(converted), 1):
        logger.info(converted[i][0])
