from typing import Union, List, Tuple, Dict, Optional, ClassVar
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    meta: Dict = field(default_factory=dict)
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
            output_sample = NLIWithOptionsSample(
                id=self.id,
                premise=self.article,
                hypothesis_options=[],
                label=self.answer)

            for h, meta in hypothesis_options:
                output_sample.hypothesis_options.append(h)
                output_sample.meta.append(meta)  # type: ignore

            return [output_sample]
        elif target == 'PureNLISample':

            hypothesis_options = self.__class__.to_nli_converter(
                [self.question] * len(self.options), self.options)

            return [
                PureNLISample(
                    id=self.id + f"_{i}",
                    premise=self.article,
                    hypothesis=h[0],
                    label=int(i == self.answer),
                    meta=h[1]) for i, h in enumerate(hypothesis_options)
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
    meta: Dict = field(default_factory=dict)
    can_convert_to: ClassVar = {
        'QuestionOptionSample', 'PureNLISample',
        'SingleQuestionSingleOptionSample'
    }

    def to(self, target: str) -> List[Sample]:
        self.check_convertible(target)

        if target == 'SingleQuestionSingleOptionSample':
            return [self]
        elif target == 'QuestionOptionSample':
            return [
                QuestionOptionSample(
                    id=self.id, question=self.question, option=self.option)
            ]
        elif target == 'PureNLISample':
            hypothesis = self.__class__.to_nli_converter([self.question],
                                                         [self.option])
            output_sample = PureNLISample(
                id=self.id,
                premise=self.article,
                hypothesis=hypothesis[0][0],
                label=self.label,
                meta=hypothesis[0][1])

            return [output_sample]
        else:
            raise ValueError(
                f"Cannot convert from {self} to {target}. Not supported")

        return []


@dataclass
class QuestionOptionSample(Sample):
    id: str
    question: str
    option: str
    meta: Dict = field(default_factory=list)


@dataclass
class NLIWithOptionsSample(Sample):
    id: str
    premise: str
    hypothesis_options: List[str]
    label: int
    meta: Dict = field(default_factory=list)


@dataclass
class PureNLISample(Sample):
    id: str
    premise: str
    hypothesis: str
    label: int
    meta: Dict = field(default_factory=dict)
