from typing import Union, List, Tuple, Dict, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import logging
from .types import (Sample, SingleQuestionSample,
                    SingleQuestionSingleOptionSample, NLIWithOptionsSample,
                    PureNLISample)
logger = logging.getLogger(__name__)


class DatasetReader(ABC):
    @abstractmethod
    def read(self, path: Union[Path, str]) -> List[Dict]:
        pass


class ConversionManager:
    def __init__(
            self,
            input_samples: List[
                Union[SingleQuestionSample, SingleQuestionSingleOptionSample]],
            batch_size: int = 10):

        if isinstance(input_samples[0], SingleQuestionSample):
            logger.warn(
                f"Batch size will be ignored because input is of type {type(SingleQuestionSample)}"
            )
            self.single_question_sample = True
        else:
            self.single_question_sample = False
        self.input_samples = input_samples
        self.batch_size = batch_size if not self.single_question_sample else 1
        self.current_batch_start = None
        self.current_batch_end = None
        self.stop = False
        self.recieved = False
        self.res = None

    def _batch(self, start: int, end: int) -> Tuple[List[str], List[str]]:
        assert end <= len(self.input_samples)
        assert start >= 0
        questions = []
        options = []

        for s in self.input_samples[start:end]:
            if self.single_question_sample:
                for option in s.options:  # type: ignore
                    questions.append(s.question)
                    options.append(option)
            else:
                questions.append(s.question)
                options.append(s.option)  # type: ignore

        return questions, options

    def __iter__(self):
        self.stop = False
        self.recieved = True
        self.current_batch_start = 0
        self.current_batch_end = self.current_batch_start + self.batch_size
        self.res = []

        return self

    def __len__(self) -> int:
        return int(math.ceil(len(self.input_samples) / self.batch_size))

    def __next__(self) -> Tuple[List[str], List[str]]:
        if self.stop:
            raise StopIteration

        if not self.recieved:
            raise RuntimeError(
                "Cannot iterate over the next batch"
                " without telling the conversion value of the current batch")

        if self.current_batch_end is None or self.current_batch_start is None:
            raise RuntimeError

        assert self.current_batch_start <= self.current_batch_end

        if self.current_batch_end <= len(self.input_samples):
            to_return = self._batch(self.current_batch_start,
                                    self.current_batch_end)

        self.recieved = False

        return to_return

    def tell(self, res: List[str]) -> None:
        if self.recieved:
            raise RuntimeError("Called tell twice without iterating")

        if self.current_batch_end is None or self.current_batch_start is None:
            raise RuntimeError("tell() called before strating to iterate")
        # store results
        current_samples = self.input_samples[self.current_batch_start:self.
                                             current_batch_end]
        # sanity checks

        # prepare iter based on type

        if not self.single_question_sample:
            assert len(current_samples) == len(res)

            for result, current_sample in zip(res, current_samples):
                self.res.append(
                    PureNLISample(
                        id=current_sample.id,
                        premise=current_sample.article,
                        hypothesis=res,
                        label=current_sample.label))
        else:
            assert len(current_samples) == 1
            assert len(current_samples[0].options) == len(res)
            self.res.append(
                NLIWithOptionsSample(
                    id=current_samples[0].id,
                    premise=current_samples[0].article,
                    hypothesis_options=res,
                    label=current_samples[0].answer))
        # prepare for next()
        self.current_batch_start += self.batch_size
        self.current_batch_end = self.current_batch_start + self.batch_size

        if self.current_batch_end > len(self.input_samples):
            self.current_batch_end = len(self.input_samples)

        if self.current_batch_start >= len(self.input_samples):
            self.stop = True
        self.recieved = True

    def converted(
            self) -> Optional[Union[List[SingleQuestionSample],
                                    List[SingleQuestionSingleOptionSample]]]:

        return self.res
