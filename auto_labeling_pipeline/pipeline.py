from typing import Dict, Any
from auto_labeling_pipeline.labels import Labels
from auto_labeling_pipeline.mappings import MappingTemplate
from auto_labeling_pipeline.models import RequestModel
from auto_labeling_pipeline.postprocessing import BasePostProcessor


def pipeline(text: str,
             request_model: RequestModel,
             mapping_template: MappingTemplate,
             post_processing: BasePostProcessor) -> Labels:
    response = request_model.send(text)
    labels = mapping_template.render(response)
    labels = post_processing.transform(labels)
    return labels


def pipeline_import(text: str,
             annotations: Dict[Any, Any],
             mapping_template: MappingTemplate,
             post_processing: BasePostProcessor) -> Labels:
    labels = mapping_template.render(annotations)
    labels = post_processing.transform(labels)
    return labels
