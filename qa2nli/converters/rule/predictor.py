from typing import Callable, List, Union, TypeVar, Tuple, Dict, Any
from ..base import Converter
from pathlib import Path
from spacy_stanfordnlp import StanfordNLPLanguage
import stanfordnlp
from copy import deepcopy
from spacy_conll import ConllFormatter
from .model import Question, AnswerSpan
from conllu import parse
from sacremoses import MosesTokenizer, MosesDetokenizer
from ..processors import PreprocessorBase, ConversionIssue, PostprocessorBase
import logging
import warnings
logger = logging.getLogger(__name__)


class RuleBasedPreprocessor(PreprocessorBase):
    """ For rule based conversion,
    entire conversion should happen in the preprocessor
    """

    def __init__(self) -> None:
        snlp = stanfordnlp.Pipeline(lang='en')  # stanfordnlp python pipeline
        self.nlp = StanfordNLPLanguage(snlp)  # spacy wraper for snlp
        conllformatter = ConllFormatter(self.nlp)
        self.nlp.add_pipe(conllformatter, last=True)
        self.detokenizer = MosesDetokenizer()
        self.vanila_preprocessor = PreprocessorBase()

    def __call__(self, q: str, o: str) -> Tuple[str, Dict]:
        if '_' in q:  # FITB. Do it and return early
            h, meta = self.vanila_preprocessor(q, o)

        return h, meta

        if o in q:
            # most likely a preprocessed FITB question
            meta = {'question': q, 'option': o}

            return q, meta

        # the old code throws UserWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q_doc = self.nlp(q)
            o_doc = self.nlp(o)
        try:
            q_conll_dict = parse(q_doc._.conll_str)[0].tokens
            o_conll_dict = parse(o_doc._.conll_str)[0].tokens
        except IndexError:
            logger.error(f"Index error on parse for {q}")
            h = q + ' ' + o
            meta: Dict[str, Any] = {
                'question': q,
                'option': o,
                'conversion_issues': [str(ConversionIssue.UNKNOWN)]
            }

            return h, meta

        rule_q = Question(deepcopy(q_conll_dict))  # type:ignore
        rule_o = AnswerSpan(deepcopy(o_conll_dict))  # type:ignore
        conversion_issues = []
        meta = {'question': q, 'option': o}

        if not rule_q.isvalid:
            conversion_issues.append(ConversionIssue.INVALID_QUESTION)

        if not rule_o.isvalid:
            conversion_issues.append(ConversionIssue.INVALID_OPTION)
        # if conversion issue is encountered just concat q + o

        if conversion_issues:
            h = q + ' ' + o
        else:
            rule_q.insert_answer_default(rule_o)
            h = self.detokenizer.detokenize(
                rule_q.format_declr(), return_str=True)
        meta['conversion_issues'] = [str(issue) for issue in conversion_issues]

        if meta['conversion_issues']:
            logger.debug(
                f"Issues {conversion_issues} encountered for {q} + {o}")

        return h, meta


class RuleBasedConverter(Converter):
    def __init__(
            self,
            model_path: Path,
            device_number: int = 0,
            preprocessor: Callable[[str, str], Tuple[str, Dict]] = None,
            postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = None):
        # take special care of preprocessor before initializing the parent class
        # it has the main logic

        if preprocessor is not None:
            logger.warning(
                f"preprocessor {preprocessor} ignored because"
                f" RuleBasedConverter does not support custom preprocesor")
        preprocessor = RuleBasedPreprocessor()
        # the parent initialize the PostProcessorBase if none is supplied.
        # That is good enough in general
        super().__init__(
            preprocessor=preprocessor, postprocessor=postprocessor)
