import pytest
from qa2nli.qa_readers.race import RaceReader, OriginalRaceSample, ANS_LETTER_TO_NUM
from qa2nli.qa_readers.reader import SingleQuestionSample, ConversionManager, PureNLISample
from qa2nli.qa_readers.writer import JSONWriter
from pathlib import Path
from dataclasses import asdict
from .converters import dummy_converter, basic_bart_converter, bart_with_spacy_converter
import json
import itertools
import tqdm
import logging
import jsonlines
logger = logging.getLogger(__file__)


@pytest.fixture
def race_dev_data():
    import json
    from dataclasses import asdict

    def read(p):
        with open(p) as f:
            return f.read()

    f = Path(__file__).absolute().parent
    high = [
        json.loads(read(f / ex))

        for ex in ['.data/RACE/dev/high/63.txt', '.data/RACE/dev/high/75.txt']
    ]
    middle = [
        json.loads(read(f / ex)) for ex in
        ['.data/RACE/dev/middle/13.txt', '.data/RACE/dev/middle/34.txt']
    ]

    return {
        ex['id']: asdict(OriginalRaceSample(**ex))

        for ex in itertools.chain(high, middle)
    }


@pytest.fixture
def race_dev_data_as_dataclasses():
    import json
    from dataclasses import asdict

    def read(p):
        with open(p) as f:
            return f.read()

    f = Path(__file__).absolute().parent
    high = [
        json.loads(read(f / ex))

        for ex in ['.data/RACE/dev/high/63.txt', '.data/RACE/dev/high/75.txt']
    ]
    middle = [
        json.loads(read(f / ex)) for ex in
        ['.data/RACE/dev/middle/13.txt', '.data/RACE/dev/middle/34.txt']
    ]

    return {
        ex['id']: (OriginalRaceSample(**ex))

        for ex in itertools.chain(high, middle)
    }


@pytest.fixture
def race_dev_data_as_dataclasses_raw():
    import json
    from dataclasses import asdict

    def read(p):
        with open(p) as f:
            return f.read()

    f = Path(__file__).absolute().parent
    high = [
        json.loads(read(f / ex))

        for ex in ['.data/RACE/dev/high/63.txt', '.data/RACE/dev/high/75.txt']
    ]
    middle = [
        json.loads(read(f / ex)) for ex in
        ['.data/RACE/dev/middle/13.txt', '.data/RACE/dev/middle/34.txt']
    ]

    return [(OriginalRaceSample(**ex)) for ex in itertools.chain(high, middle)]


@pytest.fixture
def race_data_path():
    return Path(__file__).absolute().parent / '.data/RACE/dev'


@pytest.fixture
def multirc_data_path():
    return Path(__file__).absolute().parent / '.data/multirc/dev.json'


@pytest.fixture
def boolq_data_path():
    return Path(__file__).absolute().parent / '.data/boolq/dev.jsonl'


@pytest.fixture
def boolq_data():
    with jsonlines.open(
            Path(__file__).absolute().parent / '.data/boolq/dev.jsonl') as f:
        data = list(f)

    return data
