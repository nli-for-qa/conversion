from typing import Callable, List, Union, TypeVar, Tuple, Dict, Any
from .processors import PreprocessorBase, PostprocessorBase


class Converter:
    def __init__(self,
                 *args,
                 preprocessor: Callable[[str, str], Tuple[str, Dict]] = None,
                 postprocessor: Callable[[str, Dict], Tuple[str, Dict]] = None,
                 **kwargs: None) -> None:

        if preprocessor is None:
            self.preprocessor: Callable[[str, str],
                                        Tuple[str, Dict]] = PreprocessorBase()
        else:
            self.preprocessor = preprocessor

        if postprocessor is None:
            self.postprocessor: Callable[
                [str, Dict], Tuple[str, Dict]] = PostprocessorBase()
        else:
            self.postprocessor = postprocessor

    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[Tuple[str, Dict]]:
        # check batching

        if type(question) == list and type(option) == list:
            # batched. Do nothing
            pass
        else:
            question = [question]  # type:ignore
            option = [option]  # type:ignore
        assert len(question) == len(option)
        preprocessed: List[Tuple[str, Dict]] = [
            self.preprocessor(q, o) for q, o in zip(question, option)
        ]
        preds: List[Tuple[str, Dict]] = self.apply_model(preprocessed)
        postprocessed: List[Tuple[str, Dict]] = [
            self.postprocessor(pred, meta) for pred, meta in preds
        ]

        return postprocessed

    def apply_model(self, preprocessed: List[Tuple[str, Dict]], *args: Any,
                    **kwargs: Any) -> List[Tuple[str, Dict]]:

        return preprocessed


class JustQuestionConverter(Converter):
    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[Tuple[str, Dict]]:
        # check batching

        if type(question) == list and type(option) == list:
            # batched. Do nothing
            pass
        else:
            question = [question]
            option = [option]

        return [(q, dict()) for q, o in zip(question, option)]


class ConstConverter(Converter):
    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[Tuple[str, Dict]]:
        # check batching

        if type(question) == list and type(option) == list:
            # batched. Do nothing
            pass
        else:
            question = [question]
            option = [option]

        return [('abc', dict()) for q, o in zip(question, option)]
