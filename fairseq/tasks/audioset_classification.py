import logging
import numpy as np
import torch
import csv
from sklearn import metrics as sklearn_metrics
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task, FairseqTask
from fairseq.logging import metrics

from fairseq.data.audio import audioset


logger = logging.getLogger(__name__)


@dataclass
class AudiosetClassificationConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sample_rate: int = 32000
    label_descriptors: str = "/share/rk7/home/fschmid/deployment/fairseq/examples/passt/data/label_descriptors.csv"


@register_task("audioset_classification", dataclass=AudiosetClassificationConfig)
class AudiosetClassificationTask(FairseqTask):
    """ Tasks
            - load datasets
            - iterate over datasets
            - initialize model
            - initialize criterion
            - calculate the loss
    """

    cfg: AudiosetClassificationConfig

    def __init__(
        self,
        cfg: AudiosetClassificationConfig,
    ):
        super().__init__(cfg)

        self.state.add_factory("labels", self.load_labels)

    @classmethod
    def setup_task(cls, cfg: AudiosetClassificationConfig, **kwargs):
        return cls(cfg)

    def load_labels(self):
        # load label index <-> name mapping
        labels = {}
        path = self.cfg.label_descriptors
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)

        for index in range(1, len(lines)):
            label = lines[index][2]
            labels[label] = index
        return labels

    @property
    def labels(self):
        return self.state.labels

    def load_dataset(
        self, split: str, task_cfg: AudiosetClassificationConfig = None, **kwargs
    ):
        task_cfg = task_cfg or self.cfg
        data_path = self.cfg.data
        self.datasets[split] = audioset.get_dataset(data_path, split, task_cfg.sample_rate)

    def calculate_stats(self, output, target):
        classes_num = target.shape[-1]
        stats = []

        # Class-wise statistics
        for k in range(classes_num):
            # Average precision
            avg_precision = sklearn_metrics.average_precision_score(
                target[:, k], output[:, k], average=None
            )

            dict = {
                "AP": avg_precision,
            }
            stats.append(dict)

        return stats

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if "_predictions" in logging_outputs[0]:
            metrics.log_concat_tensor(
                "_predictions",
                torch.cat([l["_predictions"].cpu() for l in logging_outputs], dim=0),
            )
            metrics.log_concat_tensor(
                "_targets",
                torch.cat([l["_targets"].cpu() for l in logging_outputs], dim=0),
            )

            def compute_stats(meters):
                if meters["_predictions"].tensor.shape[0] < 100:
                    return 0
                stats = self.calculate_stats(
                    meters["_predictions"].tensor, meters["_targets"].tensor
                )
                return np.nanmean([stat["AP"] for stat in stats])

            metrics.log_derived("mAP", compute_stats)
