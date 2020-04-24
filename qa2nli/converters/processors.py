from typing import Callable, List, Union, Optional, Dict, Tuple
import re
import spacy
import logging
import math
from enum import Enum
logger = logging.getLogger(__name__)


def remove_excess_space(inp: str) -> str:
    return ' '.join(inp.split()).strip()


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


class PreprocessorBase:
    """Override the __call__ method in inherited class to change functionallity"""

    def __call__(self, q: str, o: str) -> Tuple[str, Dict]:
        """ Very basic preprocessor which concats question and option.

            Handles fill in the black type questions.
        """

        if '_' in q:  # FITB
            h = q.replace('_', o)
        else:
            h = q + ' ' + o
        h = remove_excess_space(h)
        meta = {'question': q, 'option': o}

        return h, meta


Preprocessor = PreprocessorBase

dots = re.compile(r"[\.\'\"\?, ]{2,}[\w ]*")


def remove_dots(inp: str) -> str:
    return dots.sub('.', inp)


class ConversionIssue(Enum):
    NONE = 'none'
    TOO_SHORT = 'too_short'
    TOO_LONG = 'too_long'
    COULD_NOT_FIX = 'could_not_fix'
    INVALID_QUESTION = 'invalid_question'
    INVALID_OPTION = 'invalid_option'
    MISSING_INFORMATION = 'missing_info'
    UNGRAMTICAL_RESULT = 'ungramatical_result'
    UNKNOWN = 'unknown'

    def __str__(self) -> str:
        return self.value


class PostprocessorBase:
    def __init__(self,
                 lower_length_ratio: Optional[float] = None,
                 upper_length_ratio: float = 1.3) -> None:
        self.lower_length_ratio = lower_length_ratio
        self.upper_length_ratio = upper_length_ratio

    def __call__(self, inp: str, meta: Dict) -> Tuple[str, Dict]:
        meta.update({'conversion_issues': []})

        return inp, meta

    def _length_check(self, output: str, question: str,
                      option: str) -> ConversionIssue:
        total_ratio = (len(output) / (len(question) + len(option)))

        if total_ratio > self.upper_length_ratio:
            # too long. Cut the output

            return ConversionIssue.TOO_LONG
        elif self.lower_length_ratio is None and len(output) < len(option):
            return ConversionIssue.TOO_SHORT
        elif self.lower_length_ratio is not None:
            if total_ratio < self.lower_length_ratio:
                return ConversionIssue.TOO_SHORT

        return ConversionIssue.NONE


class Postprocessor(PostprocessorBase):
    def __init__(self,
                 sentence_splitter: str = 'period',
                 cleaner: str = None,
                 lower_length_ratio: float = None,
                 upper_length_ratio: float = 1.3) -> None:
        self.sentence_splitter = sentence_splitter

        if cleaner == 'remove_dots':
            self.cleaner: Callable[[str], str] = remove_dots
        else:
            self.cleaner = lambda x: x

        if sentence_splitter == 'spacy':
            self.spacy_nlp = get_spacy_model('en_core_web_sm')
        else:
            self.spacy_nlp = None
        super().__init__(
            lower_length_ratio=lower_length_ratio,
            upper_length_ratio=upper_length_ratio)

    def _fix_too_short(self, all_sentences: List[str],
                       meta: Dict) -> Tuple[str, bool]:
        next_ = 1
        could_not_fix = False
        current_output = all_sentences[0]
        # add sentences till legth is not too short
        max_tries = min(5, len(all_sentences))
        length_issue = LengthIssue.TOO_SHORT

        if max_tries == 1:
            could_not_fix = True

        while length_issue == LengthIssue.TOO_SHORT:

            current_output = current_output + f" {all_sentences[next_]}"
            length_issue = self._length_check(current_output, meta['question'],
                                              meta['option'])
            next_ += 1

            if next_ >= max_tries:
                could_not_fix = True

                break

        return current_output, could_not_fix

    def __call__(self, inp: str, meta: Dict) -> Tuple[str, Dict]:
        cleaned = self.cleaner(inp)

        if self.sentence_splitter == 'spacy':
            sentences = [
                s.text.strip() for s in list(self.spacy_nlp(cleaned).sents)
            ]
            first_sent = (sentences[0]).strip()

        elif self.sentence_splitter == 'period':
            sentences = cleaned.split('.')
            first_sent = sentences[0]
        meta['all_sentences'] = sentences

        output = first_sent
        issues_encountered = []
        length_issue = self._length_check(output, meta['question'],
                                          meta['option'])

        if length_issue == ConversionIssue.TOO_SHORT:
            issues_encountered.append(length_issue)
            output, could_not_fix = self._fix_too_short(sentences, meta)

            if could_not_fix:
                issues_encountered.append(ConversionIssue.COULD_NOT_FIX)
        # check again
        length_issue = self._length_check(output, meta['question'],
                                          meta['option'])

        if length_issue == ConversionIssue.TOO_LONG:
            issues_encountered.append(length_issue)
            output = output[:int(
                math.ceil(self.upper_length_ratio *
                          (len(meta['question']) + len(meta['option']))))]

        meta['conversion_issues'] = [
            str(issue) for issue in issues_encountered
        ]
        output = remove_excess_space(output)

        return output, meta
