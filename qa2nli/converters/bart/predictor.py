from typing import Callable, List, Union, TypeVar

from .model import BartSystem
from pathlib import Path
import json
from transformers import BartTokenizer
import spacy
import tqdm
import re
import logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


def apply_batch_agnostic(foo: Callable[..., T], *args) -> Union[T, List[T]]:
    """Always returns a batch"""

    if isinstance(args[0], list):
        out = [foo(*arg_sample) for arg_sample in zip(*args)]
    else:
        out = [foo(*args)]

    return out


class BartConverter:
    def __init__(self,
                 model_path: Path,
                 device_number: int = 0,
                 preprocessor: Callable[[str, str], str] = None,
                 postprocessor: Callable[[str], str] = None):

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
            self.preprocessor = lambda q, o: q + ' ' + o
        else:
            self.preprocessor = preprocessor

        if postprocessor is None:
            self.postprocessor = lambda h: h
        else:
            self.postprocessor = postprocessor

    def apply_model(
            self,
            preprocessed,
            pad_to_max_length=True,
            max_length=40,
            num_beams=1,
            do_sample=False,
            no_repeat_ngram_size=4,
            top_k=2,
    ) -> List[str]:
        inp_tensor = self.tokenizer.batch_encode_plus(
            preprocessed,
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

        return preds

    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[str]:
        preprocessed = apply_batch_agnostic(self.preprocessor, question,
                                            option)
        preds = self.apply_model(preprocessed)
        postprocessed = apply_batch_agnostic(self.postprocessor, preds)

        return postprocessed
