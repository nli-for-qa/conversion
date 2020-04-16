from typing import Callable, List, Union, Optional
import re
import spacy
import logging
logger = logging.getLogger(__name__)


def get_spacy_model(model: str) -> spacy.language.Model:
    try:
        spacy_model = spacy.load(model)
    except OSError:
        logger.warning(
            f"Spacy models '{model}' not found.  Downloading and installing.")
        spacy.cli.download(model)

        # Import the downloaded model module directly and load from there
        spacy_model_module = __import__(model)
        spacy_model = spacy_model_module.load()

    return spacy_model


class Preprocessor:
    """Override the __call__ method in inherited class to change functionallity"""

    def __call__(self, q: str, o: str) -> str:
        """ Very basic preprocessor which concats question and option.

            Handles fill in the black type questions.
        """

        if '_' in q:  # FITB
            h = q.replace('_', o)
        else:
            h = q + ' ' + o

        return h


dots = re.compile(r"[\.\'\"\?, ]{2,}[\w ]*")


def remove_dots(inp: str) -> str:
    return dots.sub('.', inp)


class Postprocessor:
    def __init__(self, sentence_splitter: str = 'period',
                 cleaner: str = None) -> None:
        self.sentence_splitter = sentence_splitter

        if cleaner == 'remove_dots':
            self.cleaner: Callable[[str], str] = remove_dots
        else:
            self.cleaner = lambda x: x

        if sentence_splitter == 'spacy':
            self.spacy_nlp = get_spacy_model('en_core_web_sm')
        else:
            self.spacy_nlp = None

    def __call__(self, inp: str) -> str:
        cleaned = self.cleaner(inp)

        if self.sentence_splitter == 'spacy':
            first_sent = (list(self.spacy_nlp(cleaned).sents)[0]).text.strip()
        elif self.sentence_splitter == 'period':
            first_sent = cleaned.split('.')[0]

        return first_sent
