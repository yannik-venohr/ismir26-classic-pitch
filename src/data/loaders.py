"""
Dataset-specific loading of audio and note annotations.
Every dataset comes with its own file layout and annotation format; this is the only
place that knows about them. Directories are set in configs/environment/*.yaml.
Note events are returned as an array of rows (start [s], end [s], MIDI pitch, ...).
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.data.utils import midi_file_to_note_events_with_pedal, midi_files_to_note_events

# annotation CSVs with columns start;end;pitch
CSV_SEMICOLON = ['bpsd', 'swd', 'krauseOrchestral', 'wrd', 'rubato', 'phoenixAnechoic',
                 'csd', 'urmp', 'dagstuhlChoirSet', 'guitarSet']
# annotation CSVs with start,end,pitch as first three columns
CSV_COMMA = ['bsqd', 'choraleBricks20']
# MIDI annotations, notes extended by the sustain pedal
MIDI = ['youChorale', 'rwc2']


def load_note_events(id, dataset, annot_dir):
    annot_dir = Path(annot_dir)
    if dataset in CSV_SEMICOLON:
        df = pd.read_csv(annot_dir / f'{id}.csv', delimiter=';')
        return np.array(df[['start', 'end', 'pitch']])
    elif dataset in CSV_COMMA:
        df = pd.read_csv(annot_dir / f'{id}.csv', delimiter=',')
        return np.array(df.iloc[:, :3])
    elif dataset == 'bach10':
        df = pd.read_csv(annot_dir / f'{id}.csv', delimiter=';')
        return np.array(df.iloc[:, :3])
    elif dataset in MIDI:
        return np.array(midi_file_to_note_events_with_pedal(annot_dir / f'{id}.mid'))
    elif dataset == 'gaps':
        return np.array(midi_file_to_note_events_with_pedal(annot_dir / f'{id}-fine-aligned.mid'))
    elif dataset == 'maestro':
        return np.array(midi_file_to_note_events_with_pedal(annot_dir / f'{id}.midi'))
    elif dataset == 'trios':
        return np.array(midi_files_to_note_events(annot_dir / id))
    elif dataset == 'musicnet':
        df = pd.read_csv(annot_dir / f'{id}.csv', delimiter=';')
        return np.array(df[['start_sec', 'end_sec', 'pitch']])
    raise ValueError(f'No annotation loader for dataset {dataset}')


def load_audio(id, dataset, audio_dir, sr):
    audio_dir = Path(audio_dir)
    if dataset == 'trios':
        path = audio_dir / id / 'mix.wav'
    elif dataset == 'bach10':
        path = audio_dir / id / f'{id}.wav'
    elif dataset == 'urmp':
        path = audio_dir / id / f'AuMix_{id}.wav'
    else:
        path = audio_dir / f'{id}.wav'
    x, sr_audio = librosa.load(path, mono=True, sr=sr)
    return x, sr_audio
