import torch.nn as nn
from dataclasses import dataclass, field

from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from models.helpers import AugmentMelSTFT

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@dataclass
class PaSSTConfig(FairseqDataclass):
    s_patchout_f: int = field(
        default=0,
        metadata={"help": "structured patchout frequency"}
    )

    s_patchout_t: int = field(
        default=0,
        metadata={"help": "structured patchout time"}
    )

    hidden_dim: int = field(
        default=1024
    )


@register_model('passt', dataclass=PaSSTConfig)
class PaSST(BaseFairseqModel):
    def __init__(self, cfg: PaSSTConfig, num_classes):
        super().__init__()
        self.cfg = cfg

        # process the config and build model here

        self.mel = AugmentMelSTFT()

        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 5),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.Conv2d(256, num_classes, 1),
            nn.AdaptiveAvgPool2d((1,1))
        )


    @classmethod
    def build_model(cls, cfg: PaSSTConfig, task: FairseqTask):
        """Build a new model instance."""

        assert hasattr(task, "labels"), f"Task {task} must have an attribute 'labels'"

        return cls(cfg, len(task.labels))

    def load_model_weights(self, state, model, cfg):
        # load model state dict
        pass

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, waveforms, labels):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch.

        # process waveform using torchaudio
        # run model forward pass
        # calculate loss

        specs = self.mel(waveforms)
        out = self.net(specs).squeeze(2).squeeze(2)
        loss = F.binary_cross_entropy(out, labels.float())

        result = {'loss': loss}

        # Return the final output state for making a prediction
        return result
