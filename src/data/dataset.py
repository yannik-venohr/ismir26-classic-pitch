from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PitchDataset(Dataset):
    """
    Splits every track into patches of context_size frames, with a hop of hop_size frames.
    A sample is the HCQT patch and the note/onset annotations of that patch.
    Tracks are loaded on demand with memory mapping, so the dataset does not need to fit into RAM.
    """
    def __init__(self, data_params, track_dir, ids, name=''):
        self.track_dir = Path(track_dir)
        self.context_size = data_params.dataset.context_size
        self.hop_size = data_params.dataset.hop_size
        self.load_onsets = data_params.dataset.load_onsets
        self.load_frames = data_params.dataset.load_frames

        # every patch is a dict with track_id, start_frame and end_frame
        self.patches = []
        n_tracks, n_frames_total = 0, 0
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._patches_of_track, ids), total=len(ids), desc=f'Preparing {name} dataset'))
        for result in results:
            if result is None:
                continue
            patches, n_frames = result
            self.patches.extend(patches)
            n_tracks += 1
            n_frames_total += n_frames

        feature_rate = data_params.hcqt.sr / data_params.hcqt.hop_length
        self.minutes_total = round(n_frames_total / feature_rate / 60)
        print(f'Added {n_tracks} of {len(ids)} tracks to the {name} dataset: '
              f'{n_frames_total / feature_rate / 60:.2f} minutes of audio divided into {len(self.patches)} '
              f'patches of {self.context_size / feature_rate:.2f} seconds each.')

    def __len__(self):
        return len(self.patches)

    def _patches_of_track(self, track_id):
        path = self.track_dir / track_id / 'hcqt.npy'
        if not path.exists():
            return None
        n_frames = np.load(path, mmap_mode='r').shape[2]
        n_patches = (n_frames - self.context_size) // self.hop_size + 1
        patches = [{
            'track_id': track_id,
            'start_frame': i * self.hop_size,
            'end_frame': i * self.hop_size + self.context_size,
        } for i in range(n_patches)]
        return patches, n_frames

    def __getitem__(self, idx_patch):
        patch = self.patches[idx_patch]
        base = self.track_dir / patch['track_id']
        s, e = patch['start_frame'], patch['end_frame']

        X = torch.from_numpy(np.load(base / 'hcqt.npy', mmap_mode='r')[:, :, s:e].copy())
        y = {}
        if self.load_frames:
            y['frame'] = torch.from_numpy(np.load(base / 'y_frame.npy', mmap_mode='r')[:, s:e].copy())
        if self.load_onsets:
            y['onset'] = torch.from_numpy(np.load(base / 'y_onset.npy', mmap_mode='r')[:, s:e].copy())
        return {'X': X, 'y': y}
