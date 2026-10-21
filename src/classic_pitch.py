import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from src.model.assembler import Assembler
from src.loss import note_loss, onset_loss_with_tolerance


def compute_frame_metrics(y_pred_logits, y_true, threshold=0.4):
    """
    Compute frame-level precision, recall, and F1 for binary predictions.

    Args:
        y_pred_logits: Predicted logits (B, F, T).
        y_true: Ground truth binary labels (B, F, T).
        threshold: Sigmoid threshold for positive classification.
    Returns:
        Dictionary with precision, recall, and f1.
    """
    y_pred = (torch.sigmoid(y_pred_logits) >= threshold).float()
    y_true = y_true.float()

    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


class ClassicPitchLightning(LightningModule):
    def __init__(self, model_config, training_config):
        """
        Args:
            model_config (dict): Configuration for the model.
            training_config (dict): Configuration for training parameters.
        """
        super().__init__()
        self.model = Assembler(model_config)
        self.loss_config = training_config.loss
        self.learning_rate = training_config.learning_rate
        self.max_steps = training_config.max_steps
        self.save_hyperparameters()

    def loss(self, y_pred, y_true):
        zero = torch.tensor(0.0, device=self.device)
        note_l, onset_l = zero, zero
        if self.loss_config.type in ["note", "combined"]:
            note_l = note_loss(
                y_true=y_true["note"],
                y_pred_logits=y_pred["note"],
                label_smoothing=self.loss_config.label_smoothing,
            )
        if self.loss_config.type in ["onset", "combined"]:
            onset_l = onset_loss_with_tolerance(
                y_true=y_true["onset"],
                y_pred_logits=y_pred["onset"],
                label_smoothing=self.loss_config.label_smoothing,
                positive_weight=self.loss_config.onset_positive_weight,
                tolerance_frames=self.loss_config.onset_tolerance_frames,
            )
        return {
            "note_loss": note_l,
            "onset_loss": onset_l,
            "total_loss": note_l + onset_l,
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["X"])
        loss_dict = self.loss(outputs, batch["y"])
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch["X"])
        loss_dict = self.loss(outputs, batch["y"])
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True)

        # frame-wise metrics on the patches as a cheap proxy for the test evaluation
        for key in ["note", "onset"]:
            if key in batch["y"]:
                metrics = compute_frame_metrics(y_pred_logits=outputs[key], y_true=batch["y"][key])
                self.log(f"val/{key}_precision", metrics["precision"], on_epoch=True)
                self.log(f"val/{key}_recall", metrics["recall"], on_epoch=True)
                self.log(f"val/{key}_f1", metrics["f1"], prog_bar=True, on_epoch=True)

        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss_dict["total_loss"]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate.initial_rate)
        if self.learning_rate.scheduler is None:
            return optimizer
        if self.learning_rate.scheduler != "cosine_annealing_with_warmup":
            raise ValueError(f"Unknown scheduler type: {self.learning_rate.scheduler}")

        # linear warmup over the first 2% of steps, then cosine decay
        warmup_steps = self.max_steps // 50
        cosine_steps = self.max_steps - warmup_steps
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")
        for logger in self.loggers:
            logger.log_hyperparams({"model/num_params": num_params})

    def predict_full_track(self, hcqt, max_frames=1000, overlap=10, batch_size=32):
        """
        Predict a full track by splitting the HCQT into overlapping chunks.
        Args:
            hcqt: HCQT of shape (C, F, T)
        Returns:
            dict with 'note' and 'onset' activations in [0, 1], each of shape (72, T)
        """
        hcqt = torch.as_tensor(hcqt, dtype=torch.float32, device=self.device)

        T = hcqt.shape[-1]
        step = max_frames - overlap

        # 1. Collect chunk boundaries
        starts = list(range(0, T, step))

        # 2. Pad input so all chunks are max_frames long
        pad_needed = max(0, starts[-1] + max_frames - T)
        if pad_needed:
            hcqt = torch.nn.functional.pad(hcqt, (0, pad_needed))

        # 3. Stack all chunks → (N, C, F, max_frames)
        chunks = torch.stack([hcqt[..., s:s + max_frames] for s in starts])

        # 4. Batched inference
        self.model.eval()
        all_outputs = {}
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                out = self.model(chunks[i:i + batch_size])
                for k, v in out.items():
                    all_outputs.setdefault(k, []).append(v.float().cpu())
        all_outputs = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}

        # 5. Stitch: trim overlap and reassemble
        results = {}
        for k, v in all_outputs.items():
            pieces = []
            for j, s in enumerate(starts):
                chunk_out = v[j]
                if j > 0:
                    chunk_out = chunk_out[..., overlap:]
                pieces.append(chunk_out)
            full = torch.cat(pieces, dim=-1)[..., :T]
            results[k] = torch.sigmoid(full).numpy()
        return results
