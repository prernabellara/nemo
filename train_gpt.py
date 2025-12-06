import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf
import pytorch_lightning as pl

if __name__ == "__main__":
    config = OmegaConf.load("nemo_examples/config/finetune_gpt.yaml")

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
    )

    model = MegatronGPTModel(cfg=config.model, trainer=trainer)
    trainer.fit(model)
