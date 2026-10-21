import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/gudgud96/basic-pitch-torch/tree/main
class BasicPitch(nn.Module):
    def __init__(
        self,
        num_in_channels=6,
    ) -> None:
        super().__init__()
        self.bn_layer = nn.BatchNorm2d(1, eps=0.001)
        self.conv_contour = nn.Sequential(
            # NOTE: in the original implementation, this part of the network should be dangling...
            # nn.Conv2d(num_in_channels, 32, kernel_size=5, padding="same"),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Conv2d(num_in_channels, 8, kernel_size=(3, 3 * 13), padding="same"),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, padding="same"),
            #nn.Sigmoid()
        )
        self.conv_note = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(7, 3), padding="same"),
            #nn.Sigmoid()
        )
        self.conv_onset_pre = nn.Sequential(
            nn.Conv2d(num_in_channels, 32, kernel_size=5, stride=(1, 3)),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(),
        )
        self.conv_onset_post = nn.Sequential(
            nn.Conv2d(32 + 1, 1, kernel_size=3, stride=1, padding="same"),
            #nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, F, T) -> (B, H, T, F)
        x = x.permute(0, 1, 3, 2)
        x_contour = self.conv_contour(x) # (batch, number of frequency bins, number of time frames)

        # for strided conv, padding is different between PyTorch and TensorFlow
        # we use this equation: pad = [ (stride * (output-1)) - input + kernel ] / 2
        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 7) / 2, (3 * 87 - 264 + 7) / 2) = (3, 2)
        # F.pad process from the last dimension, so it's (2, 2, 3, 3)
        x_contour_for_note = F.pad(x_contour, (2,2,3,3))
        x_note = self.conv_note(x_contour_for_note)

        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 5) / 2, (3 * 87 - 264 + 5) / 2) = (2, 1)
        # F.pad process from the last dimension, so it's (1, 1, 2, 2)
        x_for_onset = F.pad(x, (1,1,2,2))
        x_onset_pre = self.conv_onset_pre(x_for_onset)
        x_onset_pre = torch.cat([x_note, x_onset_pre], dim=1)
        x_onset = self.conv_onset_post(x_onset_pre)

        # squeeze channel dim: (B, 1, T, F) -> (B, T, F) -> (B, F, T)
        outputs = {"onset": x_onset.squeeze(1).permute(0, 2, 1), "note": x_note.squeeze(1).permute(0, 2, 1)}

        return outputs
