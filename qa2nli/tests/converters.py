from typing import List, Tuple
import pytest
from qa2nli.converters.bart.predictor import BartConverter
from qa2nli.converters.processors import Preprocessor, Postprocessor
from pathlib import Path

bart_model_path = Path(__file__).parent.parent.parent / \
    '.models/bart_gsc4_1/epoch=1.ckpt'


@pytest.fixture
def dummy_converter():
    def converter(qs: List[str], os: List[str]) -> List[Tuple]:
        return [(q + ' ' + o, dict()) for q, o in zip(qs, os)]

    return converter


@pytest.fixture
def basic_bart_converter():
    return BartConverter(bart_model_path, device_number=-1)


@pytest.fixture
def bart_with_spacy_converter():
    return BartConverter(
        bart_model_path,
        device_number=-1,
        preprocessor=Preprocessor(),
        postprocessor=Postprocessor())
