import torch
import torch.nn as nn
from src.model.hcqt_encoder import HCQTEncoder
from src.model.onset import OnsetModule
from src.model.mpe import MPE
from src.model.basic_pitch import BasicPitch


class Assembler(nn.Module):
    """
    Builds the model from the model config. Three assembly methods:
        classic_pitch: HCQTEncoder for frames, OnsetModule on its features for onsets
        just-mpe:      a single multi-pitch model (DRCNN), its output is used for frames and onsets
        basic_pitch:   the Basic Pitch baseline
    Input:  HCQT (B, C, F, T)
    Output: dict with 'frame' and 'onset' logits (B, 72, T)
    """
    def __init__(self, model_params: dict) -> None:
        super().__init__()
        self.log_compress_input = model_params.get("log_compress_input")
        self.assembly_method = model_params["assembly_method"]

        if self.assembly_method == "basic_pitch":
            self.basic_pitch = BasicPitch(num_in_channels=model_params["n_chan_input"])
        elif self.assembly_method in ["classic_pitch", "just-mpe"]:
            cfg = model_params["mpe_encoder"]
            if cfg["type"] == "HCQTEncoder":
                self.mpe_encoder = HCQTEncoder(**cfg)
            elif cfg["type"] == "MPE":
                self.mpe_encoder = MPE(**cfg)
            else:
                raise ValueError(f"Unknown mpe_encoder type: {cfg['type']}")
            if self.assembly_method == "classic_pitch":
                self.onset_post = OnsetModule(**model_params["onset_post"])
        else:
            raise ValueError(f"Unknown assembly_method: {self.assembly_method}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.log_compress_input:
            x = torch.log1p(x * 1000)

        if self.assembly_method == "basic_pitch":
            return self.basic_pitch(x)

        if self.assembly_method == "just-mpe":
            x_note = self.mpe_encoder(x)                     # (B, 1, F, T)
            x_onset = x_note.clone()                         # (B, 1, F, T)
        elif self.assembly_method == "classic_pitch":
            features = self.mpe_encoder.forward_features(x)  # (B, C, F, T)
            x_note = self.mpe_encoder.head(features)         # (B, 1, F, T)
            x_onset = self.onset_post(features)              # (B, 1, F, T)

        return {
            "frame": x_note.squeeze(1),   # (B, F, T)
            "onset": x_onset.squeeze(1), # (B, F, T)
        }
