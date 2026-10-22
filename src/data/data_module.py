import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PitchDataset
from src.data.loaders import load_audio, load_note_events
from src.data.utils import audio_to_hcqt, note_events_to_binary_mpe_array, note_events_to_onset_array


def feature_config(data_params):
    """The parameters that determine the precomputed features."""
    full_config = OmegaConf.to_container(data_params, resolve=True)
    return {k: full_config[k] for k in ['annotations', 'hcqt']}


def feature_dir_for_config(data_params):
    """Features are stored in a subdirectory of feature_dir named after the hash of the feature config."""
    config_str = json.dumps(feature_config(data_params), sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return Path(data_params.feature_dir) / config_hash


class PitchDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the automatic music transcription task.
    All datasets are converted into a homogeneous format once: for every track, a directory
    with hcqt.npy, y_frame.npy, y_onset.npy and note_events.npy is saved to the feature dir.
    Which tracks belong to which split is defined by the boolean columns of the splits file.
    Args:
        data_params: OmegaConf object containing the data parameters (see configs/data/default.yaml)
        recompute_features: recompute the features even if they already exist
    """

    def __init__(self, data_params, recompute_features=False):
        super().__init__()
        self.data_params = data_params

        self.feature_dir = feature_dir_for_config(data_params)
        if recompute_features and self.feature_dir.exists():
            shutil.rmtree(self.feature_dir)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        with open(self.feature_dir / 'config.json', 'w') as f:
            json.dump(feature_config(data_params), f, indent=2)
        print(f"Directory for precomputed features: {self.feature_dir}")

        # optionally, train from a copy on fast local storage
        if data_params.node_cache_dir:
            self.track_dir = Path(data_params.node_cache_dir) / self.feature_dir.name
        else:
            self.track_dir = self.feature_dir

        self.splits_df = pd.read_csv(data_params.splits_file)  # id, dataset and a boolean column per split
        self.train_ids, self.val_ids, self.test_ids = self._init_splits()

        self.batch_size = data_params.dataloader.batch_size
        self.num_workers = data_params.dataloader.num_workers
        self.persistent_workers = data_params.dataloader.persistent_workers and self.num_workers > 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._prepare_data(self.train_ids + self.val_ids)
            self._copy_to_node_cache(self.train_ids + self.val_ids)
            self.train_dataset = PitchDataset(self.data_params, self.track_dir, self.train_ids, name='train')
            self.train_minutes_total = self.train_dataset.minutes_total
            self.val_dataset = PitchDataset(self.data_params, self.track_dir, self.val_ids, name='val')

        if stage == 'test' or stage is None:
            # no DataLoader needed, the test set is predicted track by track in src/eval/evaluate.py
            self._prepare_data(self.test_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def _init_splits(self):
        splits = self.data_params.dataset.splits
        train_splits, val_splits, test_splits = list(splits.train), list(splits.val), list(splits.test)
        train_omit_splits = list(splits.get('train_omit') or [])
        print(f"Train splits: {train_splits} (omitting {train_omit_splits}), Val splits: {val_splits}, Test splits: {test_splits}")
        missing = set(train_splits + train_omit_splits + val_splits + test_splits) - set(self.splits_df.columns)
        assert not missing, f"Splits {missing} are not defined in the splits file"

        def ids_in(columns):
            return self.splits_df.loc[self.splits_df[columns].any(axis=1), 'id'].tolist()

        train_ids = ids_in(train_splits)
        if train_omit_splits:
            omit_ids = set(ids_in(train_omit_splits))
            train_ids = [id for id in train_ids if id not in omit_ids]
        val_ids = ids_in(val_splits)
        test_ids = ids_in(test_splits)

        if not self.data_params.dataset.accept_overlap:
            assert not set(train_ids) & set(val_ids), "Overlap between train and val"
            assert not set(train_ids) & set(test_ids), "Overlap between train and test"
            assert not set(val_ids) & set(test_ids), "Overlap between val and test"

        print(f"Number of tracks per split: Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        return train_ids, val_ids, test_ids

    def _prepare_data(self, ids):
        """Compute HCQT and annotations for all tracks that are not in the feature dir yet."""
        todo = [id for id in ids if not (self.feature_dir / id).exists()]
        print(f"Preparing hcqt and annotations for {len(todo)} of {len(ids)} tracks")
        failed = []
        for id in tqdm(todo, desc='Computing features'):
            try:
                self._prepare_track(id)
            except Exception as e:
                print(f'Error with {id}: {e}')
                failed.append(id)
        if failed:
            print(f'Could not prepare {len(failed)} tracks, they will be skipped.')

    def _prepare_track(self, id):
        dataset = self.splits_df.loc[self.splits_df['id'] == id, 'dataset'].values[0]
        dirs = self.data_params.dataset_dirs.get(dataset)
        if dirs is None:
            raise ValueError(f'no directories configured for dataset {dataset}')
        annotations = self.data_params.annotations

        x, sr_audio = load_audio(id, dataset, dirs.audio_dir, sr=self.data_params.hcqt.sr)
        hcqt, times, _ = audio_to_hcqt(x, sr_audio, self.data_params.hcqt)

        note_events = load_note_events(id, dataset, dirs.annot_dir)
        frames_array = note_events_to_binary_mpe_array(note_events, times, annotations.min_note, annotations.out_bins)
        onsets_array = note_events_to_onset_array(note_events, times, annotations.min_note, annotations.out_bins)

        # write to a temporary dir first, so an interrupted run leaves no half-written track
        track_dir = self.feature_dir / id
        tmp_dir = track_dir.with_name(track_dir.name + '.tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / 'hcqt.npy', hcqt.astype(np.float32))
        np.save(tmp_dir / 'y_frame.npy', frames_array.astype(np.float32))
        np.save(tmp_dir / 'y_onset.npy', onsets_array.astype(np.float32))
        np.save(tmp_dir / 'note_events.npy', note_events)
        tmp_dir.rename(track_dir)

    def _copy_to_node_cache(self, ids):
        """Copy the features of the given tracks to node_cache_dir, if one is set."""
        if self.track_dir == self.feature_dir:
            return

        def copy(id):
            if (self.feature_dir / id).exists():
                shutil.copytree(self.feature_dir / id, self.track_dir / id, dirs_exist_ok=True)

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(copy, ids), total=len(ids), desc=f'Copying features to {self.track_dir}'))
