from typing import Callable, List, Union, TypeVar, Tuple, Dict, Any

from .model import BartSystem
from ..base import Converter
from pathlib import Path
import json
from transformers import BartTokenizer
import spacy
import tqdm
import re
import logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


def apply_batch_agnostic(foo: Callable, *args: Any) -> List[Tuple]:
    """Always returns a batch"""

    if isinstance(args[0], list):  # batched
        out = [foo(*arg_sample) for arg_sample in zip(*args)]
    else:
        out = [foo(*args)]

    return out


class BartConverter(Converter):
    def __init__(
            self,
            model_path: Path,
            device_number: int = 0,
            preprocessor: Callable[[str, str], Tuple[str, Dict]] = None,
            postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = None):

        if device_number > -1:
            self.device = f'cuda:{device_number}'
        else:
            self.device = 'cpu'
        logger.info(f"Loading model from {model_path} on device {self.device}")
        self.model = BartSystem.load_from_checkpoint(
            model_path, map_location=self.device)
        self.model.model.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

        if preprocessor is None:
            self.preprocessor: Callable[
                [str, str], Tuple[str, Dict]] = lambda q, o: (q + ' ' + o, {})
        else:
            self.preprocessor = preprocessor

        if postprocessor is None:
            self.postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = \
                lambda h, d: (h, d)
        else:
            self.postprocessor = postprocessor

    def apply_model(
            self,
            preprocessed: List[Tuple[str, Dict]],
            pad_to_max_length: bool = True,
            max_length: int = 40,
            num_beams: int = 1,
            do_sample: bool = False,
            no_repeat_ngram_size: int = 4,
            top_k: int = 2,
    ) -> List[Tuple[str, Dict]]:
        inp = [t[0] for t in preprocessed]
        inp_tensor = self.tokenizer.batch_encode_plus(
            inp,
            pad_to_max_length=True,
            max_length=max_length,
            return_tensors='pt')
        out_tensor = self.model.model.generate(
            inp_tensor["input_ids"].to(device=self.device),
            attention_mask=inp_tensor["attention_mask"].to(device=self.device),
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            max_length=max_length,
            early_stopping=True,
        ).cpu()
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for g in out_tensor
        ]
        # pack again
        output = [(pred, meta) for pred, (_, meta) in zip(preds, preprocessed)]

        return output

    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[Tuple[str, Dict]]:
        # check batching

        if type(question) == list and type(option) == list:
            # batched. Do nothing
            pass
        else:
            question = [question]
            option = [option]
        assert len(question) == len(option)
        preprocessed: List[Tuple[str, Dict]] = [
            self.preprocessor(q, o) for q, o in zip(question, option)
        ]
        preds: List[Tuple[str, Dict]] = self.apply_model(preprocessed)
        postprocessed: List[Tuple[str, Dict]] = [
            self.postprocessor(pred, meta) for pred, meta in preds
        ]

        return postprocessed


class BartLikeConst(BartConverter):
    def __init__(
            self,
            model_path: Path,
            device_number: int = 0,
            preprocessor: Callable[[str, str], Tuple[str, Dict]] = None,
            postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = None):

        if device_number > -1:
            self.device = f'cuda:{device_number}'
        else:
            self.device = 'cpu'
        logger.info(f"Loading model from {model_path} on device {self.device}")
        self.model = None
        self.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

        if preprocessor is None:
            self.preprocessor: Callable[
                [str, str], Tuple[str, Dict]] = lambda q, o: (q + ' ' + o, {})
        else:
            self.preprocessor = preprocessor

        if postprocessor is None:
            self.postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = \
                lambda h, d: (h, d)
        else:
            self.postprocessor = postprocessor

    def apply_model(self, preprocessed: List[Tuple[str, Dict]], *args,
                    **kwargs):

        return [('abc. abc. abc. abc.', meta) for (_, meta) in preprocessed]
