import pytest
from qa2nli.converters.bart.predictor import BartConverter
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

bart_model_path = Path(__file__).parent.parent.parent / \
    '.models/bart_test/epoch=1.ckpt'


@pytest.fixture
def sample1():
    inputs = ([
        "What was his name?", "How well did she play?",
        "Can the mountain be climbed from the other side?"
    ], [
        "Justin", "She played well but was a little inexperienced at the game",
        "Yes"
    ])
    outputs = [
        "His name was Justin.",
        "She played well but was a little inexperienced",
        "The mountain can be climbed from the other side"
    ]

    return inputs, outputs


def test_BartConverter_io(sample1, caplog):
    with caplog.at_level(logging.INFO):
        logger.info("Loading model")
        model = BartConverter(bart_model_path, device_number=-1)
        converted = model(*sample1[0])

        for i, ex in enumerate(converted):
            logger.info(f"Converted {i}: {ex}")

            if i > 2:
                break
