import torch.nn as nn

class MPE(nn.Module):  # DRCNN baseline
    """
    Input shape: (batch_size, harmonics, freq, time)
    Args:
        freq_bins_in:     Number of input frequency bins (e.g. 216 for 6 octaves with 12 bins each)
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """
    def __init__(self, freq_bins_in=216, n_chan_input=6, n_chan_layers=[20,20,10,1], n_prefilt_layers=5, a_lrelu=0.3, p_dropout=0.2, use_MaxPool=True, **kwargs):
        super(MPE, self).__init__()
        ch_in = n_chan_input
        ch_layers = n_chan_layers

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[ch_in, freq_bins_in])

        # Prefiltering
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_layers[0], kernel_size=(15,15), padding=15 // 2, stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)) if use_MaxPool else nn.Identity(),
            nn.Dropout(p=p_dropout)
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_list = nn.ModuleList(
            nn.Sequential(
                            nn.Conv2d(in_channels=ch_layers[0], out_channels=ch_layers[0], kernel_size=(15,15), padding=15 // 2, stride=(1,1)),
                            nn.LeakyReLU(negative_slope=a_lrelu),
                            nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)),
                            nn.Dropout(p=p_dropout)
                        )
            for _ in range(n_prefilt_layers - 1)
        )

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=ch_layers[0], out_channels=ch_layers[1], kernel_size=(3,3), padding=(0,1), stride=(3,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(1,13), stride=(1,1), padding=(0,6)) if use_MaxPool else nn.Identity(),
            nn.Dropout(p=p_dropout)
        )

        # Channel reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=ch_layers[1], out_channels=ch_layers[2], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Channel reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=ch_layers[2], out_channels=ch_layers[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=ch_layers[3], out_channels=1, kernel_size=(1,1), padding=(0,0), stride=(1,1)),
        )

    def forward(self, x):
        # in dims (batch_size, harmonics, freq, time)
        x = x.permute(0, 3, 1, 2)
        x = self.layernorm(x) # shape (batch_size, time, harmonics, freq)
        x = x.permute(0, 2, 3, 1)

        x = self.conv1(x)
        for p in range(0,self.n_prefilt_layers-1):
            prefilt_layer = self.prefilt_list[p]
            x_new = prefilt_layer(x)
            x = x_new + x
        x = self.conv2(x)
        x = self.conv3(x)
        y_pred = self.conv4(x)
        return y_pred
