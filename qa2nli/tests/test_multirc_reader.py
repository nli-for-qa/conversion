import pytest
from typing import List
from qa2nli.qa_readers.multirc import MultircReader, OriginalMultircSample
from qa2nli.qa_readers.types import SingleQuestionSingleOptionSample
from qa2nli.qa_readers.reader import SingleQuestionSample, ConversionManager, PureNLISample
from qa2nli.qa_readers.writer import JSONWriter
from pathlib import Path
from dataclasses import asdict
from .data import (multirc_data_path)
from .converters import bart_with_spacy_converter
import json
import itertools
import tqdm
import logging
logger = logging.getLogger(__file__)


def test_multirc_reader(multirc_data_path):
    reader = MultircReader()
    data = reader.read(multirc_data_path)

    for i in range(3):
        logger.info(data[i])


def test_multirc_to_PureNLI_bart_with_Prepostprocessors(
        multirc_data_path, bart_with_spacy_converter):
    reader = MultircReader()
    SingleQuestionSingleOptionSample.to_nli_converter = bart_with_spacy_converter
    data: List[SingleQuestionSingleOptionSample] = reader.read(
        multirc_data_path)

    converted = [sample.to('PureNLISample') for sample in data]

    for i in range(0, len(converted), 3):
        logger.info(converted[i][0])
