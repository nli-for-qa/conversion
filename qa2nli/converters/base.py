from typing import Callable, List, Union, TypeVar, Tuple, Dict, Any


class Converter:
    def __init__(self, *args: None, **kwargs: None) -> None:
        pass

    def __call__(self, question: Union[str, List[str]],
                 option: Union[str, List[str]]) -> List[Tuple[str, Dict]]:
        # check batching

        if type(question) == list and type(option) == list:
            # batched. Do nothing
            pass
        else:
            question = [question]
            option = [option]

        return [(q + ' ' + o, dict()) for q, o in zip(question, option)]


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
