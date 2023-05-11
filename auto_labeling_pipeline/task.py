import abc
from typing import Type

from auto_labeling_pipeline.labels import ClassificationLabels, Labels, Seq2seqLabels, SequenceLabels, SequenceAndRelLabels


class Task(abc.ABC):
    label_collection: Type[Labels]


class GenericTask(Task):
    label_collection = Labels


class DocumentClassification(Task):
    label_collection = ClassificationLabels


class SequenceLabeling(Task):
    label_collection = SequenceLabels


class SequenceAndRelationLabeling(Task):
    label_collection = SequenceAndRelLabels


class Seq2seq(Task):
    label_collection = Seq2seqLabels


class ImageClassification(Task):
    label_collection = ClassificationLabels


class SpeechToText(Task):
    label_collection = Seq2seqLabels


class TaskFactory:

    @classmethod
    def create(cls, task_name: str) -> Type[Task]:
        return {
            'DocumentClassification': DocumentClassification,
            'SequenceLabeling': SequenceLabeling,
            'SequenceAndRelationLabeling': SequenceAndRelationLabeling,
            'Seq2seq': Seq2seq,
            'ImageClassification': ImageClassification,
            'Speech2text': SpeechToText
        }.get(task_name, GenericTask)
