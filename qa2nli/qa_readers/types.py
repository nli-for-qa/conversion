from typing import Union, List, Tuple, Dict, Optional, ClassVar
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    id: str
    can_convert_to: ClassVar = set()
    to_nli_converter: ClassVar = None

    def check_convertible(self, target: str) -> None:
        if target not in self.can_convert_to:
            raise ValueError(f"Cannot convert {self} to {target}")

    def to(self, target: str) -> List["Sample"]:
        raise NotImplementedError


@dataclass
class SingleQuestionSample(Sample):
    id: str  # complete id till question ex: high123.txt_1
    article: str
    question: str
    options: List[str]
    answer: int
    can_convert_to: ClassVar = {
        'SingleQuestionSample', 'QuestionOptionSample', 'NLIWithOptionsSample',
        'PureNLISample'
    }

    def check_convertible(self, target: str) -> None:
        super().check_convertible(target)

        if 'NLI' in target and self.to_nli_converter is None:
            raise ValueError(
                f"Cannot convert {self} to {target} because to_nli_converter is {self.to_nli_converter}"
            )

    def to(self, target: str) -> List[Sample]:
        self.check_convertible(target)

        if target == 'SingleQuestionSample':
            return [self]
        elif target == 'QuestionOptionSample':
            return [
                QuestionOptionSample(
                    id=self.id + f"_{i}", question=self.question, option=opt)

                for i, opt in enumerate(self.options)
            ]
        elif target == 'NLIWithOptionsSample':
            hypothesis_options = self.__class__.to_nli_converter(
                [self.question] * len(self.options), self.options)

            return [
                NLIWithOptionsSample(
                    id=self.id,
                    premise=self.article,
                    hypothesis_options=hypothesis_options,
                    label=self.answer)
            ]
        elif target == 'PureNLISample':

            hypothesis_options = self.__class__.to_nli_converter(
                [self.question] * len(self.options), self.options)

            return [
                PureNLISample(
                    id=self.id + f"_{i}",
                    premise=self.article,
                    hypothesis=h,
                    label=int(i == self.answer))

                for i, h in enumerate(hypothesis_options)
            ]
        else:
            raise ValueError(
                f"Cannot convert from {self} to {target}. Not supported")

        return []


@dataclass
class SingleQuestionSingleOptionSample(Sample):
    id: str  # complete id till question ex: high123.txt_1_A
    article: str
    question: str
    option: str
    label: int


@dataclass
class QuestionOptionSample(Sample):
    id: str
    question: str
    option: str


@dataclass
class NLIWithOptionsSample(Sample):
    id: str
    premise: str
    hypothesis_options: List[str]
    label: int


@dataclass
class PureNLISample(Sample):
    id: str
    premise: str
    hypothesis: str
    label: int
