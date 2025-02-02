import abc
import operator
from typing import Dict, Iterable, List, Optional, Type

from auto_labeling_pipeline.label import ClassificationLabel, Label, Seq2seqLabel, SequenceLabel, SequenceRelAndTraitLabel


class Labels(abc.ABC):
    label_class: Type[Label]

    def __init__(self, labels: List[Dict]):
        self.labels = [self.label_class(**label) for label in labels]

    def filter_by_name(self, vocabulary: Optional[Iterable[str]] = None) -> 'Labels':
        if not vocabulary:
            return self
        labels = [label.dict() for label in self.labels if label.included(vocabulary)]
        return self.__class__(labels)

    def replace_label(self, mapping: Optional[Dict[str, str]] = None) -> 'Labels':
        if not mapping:
            return self
        labels = [label.replace(mapping).dict() for label in self.labels]
        return self.__class__(labels)

    @abc.abstractmethod
    def remove_overlapping(self) -> 'Labels':
        raise NotImplementedError

    def merge(self, others: 'Labels') -> 'Labels':
        self.labels.extend(others.labels)
        return self.remove_overlapping()

    def dict(self) -> List[dict]:
        return [label.dict() for label in self.labels]


class ClassificationLabels(Labels):
    label_class = ClassificationLabel

    def remove_overlapping(self) -> 'Labels':
        return self.__class__([label.dict() for label in set(self.labels)])


class SequenceLabels(Labels):
    label_class = SequenceLabel

    def remove_overlapping(self) -> 'Labels':
        target = self.label_class(start_offset=0, end_offset=0, label='')
        labels = sorted(self.labels, key=operator.attrgetter('start_offset'))
        _labels = []
        for label in labels:
            if label.overlap_with(target):
                continue
            target = label  # type: ignore
            _labels.append(label)
        return self.__class__([label.dict() for label in _labels])


# Vostok - Start
class SequenceRelAndTraitLabels(Labels):
    label_class = SequenceRelAndTraitLabel

    def remove_overlapping(self) -> 'Labels':
        return self
        # target = self.label_class(start_offset=0, end_offset=0, label='')
        # labels = sorted(self.labels, key=operator.attrgetter('start_offset'))
        # _labels = []
        # for label in labels:
        #     if label.overlap_with(target):
        #         continue
        #     target = label  # type: ignore
        #     _labels.append(label)
        # return self.__class__([label.dict() for label in _labels])
# Vostok - End


class Seq2seqLabels(Labels):
    label_class = Seq2seqLabel

    def filter_by_name(self, vocabulary: Optional[Iterable[str]] = None) -> Labels:
        return self

    def replace_label(self, mapping: Optional[Dict[str, str]] = None) -> Labels:
        return self

    def remove_overlapping(self) -> 'Labels':
        return self.__class__([label.dict() for label in set(self.labels)])
