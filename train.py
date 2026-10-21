import shutil
from pathlib import Path

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from src.classic_pitch import ClassicPitchLightning
from src.data.data_module import PitchDataModule
from src.eval.evaluate import evaluate_test_set


def create_loggers(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    loggers = []
    if "csv" in cfg.logging.logger:
        csv_logger = CSVLogger(save_dir=cfg.run_dir, name="logs")
        csv_logger.log_hyperparams(cfg_dict)
        loggers.append(csv_logger)
    if "wandb" in cfg.logging.logger:
        from lightning.pytorch.loggers import WandbLogger
        # remember the wandb run id, so that resumed runs continue the same wandb run
        run_id_file = Path(cfg.run_dir) / "wandb_id.txt"
        run_id = run_id_file.read_text().strip() if run_id_file.exists() else None
        wandb_logger = WandbLogger(project=cfg.logging.project, name=cfg.logging.name, save_dir=cfg.run_dir,
                                   notes=cfg.logging.notes, id=run_id, resume="allow")
        wandb_logger.experiment.config.update(cfg_dict, allow_val_change=True)
        run_id_file.write_text(wandb_logger.experiment.id)
        loggers.append(wandb_logger)
    return loggers


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_dir = Path(cfg.run_dir)

    resume_ckpt = None
    if cfg.train == "resume":
        resume_ckpt = run_dir / "last.ckpt"
        print(f"Resuming training from checkpoint: {resume_ckpt}")
    elif cfg.train == "from_scratch":
        # remove results of a previous run with the same name (hydra/ holds the config of this run)
        for path in run_dir.glob("*"):
            if path.name != "hydra":
                print(f"Deleting old results: {path}")
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    elif cfg.train:
        raise ValueError(f"Unknown train mode: {cfg.train}")

    loggers = create_loggers(cfg)
    if cfg.seed:
        seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    datamodule = PitchDataModule(cfg.data, recompute_features=cfg.recompute_features)

    if cfg.train:
        model = ClassicPitchLightning(model_config=cfg.model, training_config=cfg.training)
        checkpoint_best = ModelCheckpoint(dirpath=cfg.run_dir, monitor='val/total_loss', filename=cfg.training.filename,
                                          save_top_k=1, mode='min', save_last=True)
        trainer = Trainer(default_root_dir=cfg.run_dir, logger=loggers, max_steps=cfg.training.max_steps,
                          precision=cfg.training.precision, callbacks=[checkpoint_best], accelerator=cfg.device,
                          gradient_clip_val=cfg.training.gradient_clip_val)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt)
        for logger in loggers:
            logger.log_hyperparams({'train_minutes': datamodule.train_minutes_total})

    if cfg.eval:
        if cfg.train:  # best checkpoint of this training run
            checkpoint = checkpoint_best.best_model_path
        elif cfg.checkpoint_eval:
            checkpoint = Path(cfg.checkpoint_eval)
        else:
            checkpoint = run_dir / f"{cfg.training.filename}.ckpt"
        print(f"Loading model from checkpoint: {checkpoint}")

        model = ClassicPitchLightning(model_config=cfg.model, training_config=cfg.training)
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
        model.to(torch.device(cfg.device)).eval()

        datamodule.setup(stage='test')
        measures = evaluate_test_set(cfg, model, datamodule)
        for logger in loggers:
            logger.log_metrics(measures)
            logger.finalize("success")

    print("done, bye!")


if __name__ == '__main__':
    main()
