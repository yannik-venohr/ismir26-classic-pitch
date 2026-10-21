import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.utils import get_hcqt_times, note_events_to_binary_mpe_array, note_events_to_midi
from src.eval.metrics import eval_frame_level, eval_note_level
from src.postprocess.postprocess import output_to_note_events


def evaluate_note_events(args):
    """
    Note-level measures (AMT_*) and frame-level measures (AMT_FRAME_*) of predicted note events for one track.
    Args (one tuple, so it can be used with Pool.imap):
        predicted_note_events: list of (start, end, pitch, ...)
        track_dir: feature directory of the track with note_events.npy and y_note.npy
        hcqt_cfg: HCQT parameters (frame times)
        min_note: MIDI pitch of the lowest pitch bin
    """
    predicted_note_events, track_dir, hcqt_cfg, min_note = args
    note_events_gt = np.load(track_dir / 'note_events.npy')
    y_note = np.load(track_dir / 'y_note.npy')
    times = get_hcqt_times(hcqt_cfg, y_note.shape[1])

    measures = {}
    note_level = eval_note_level(note_events_gt, predicted_note_events)
    measures.update({f'AMT_{k}': v for k, v in note_level.items()})

    y_pred = note_events_to_binary_mpe_array(predicted_note_events, times, min_note=min_note, n_pitch_bins=y_note.shape[0])
    frame_level = eval_frame_level(y_note, y_pred)
    measures.update({f'AMT_FRAME_{k}': v for k, v in frame_level.items()})
    return measures


def evaluate_test_set(cfg, model, datamodule):
    """
    Predict all tracks of the test set, save the predictions as MIDI files and evaluate them.

    Measures:
        MPE_*:       frame-level, on the raw note activations (multi-pitch estimation)
        AMT_*:       note-level, on the decoded note events (mir_eval, onset tolerance in ms as prefix)
        AMT_FRAME_*: frame-level, on the decoded note events

    Args:
        cfg (DictConfig): Hydra configuration
        model (ClassicPitchLightning): model to evaluate
        datamodule (PitchDataModule): data module containing the test set
    Returns:
        mean_measures (dict): mean of every measure over all tracks
    """
    predictions_dir = Path(cfg.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    hcqt_cfg = cfg.data.hcqt

    all_measures = {}
    pending_note_evals = {}  # note-level eval is parallelized after the prediction loop

    for test_id in tqdm(datamodule.test_ids, desc="Predicting test set"):
        track_dir = datamodule.feature_dir / test_id
        if not track_dir.exists():
            print(f"Skipping {test_id}: no features")
            continue

        hcqt = np.load(track_dir / 'hcqt.npy')
        prediction = model.predict_full_track(hcqt, max_frames=cfg.model.predict_max_frames)  # {'note': , 'onset': }
        times = get_hcqt_times(hcqt_cfg, hcqt.shape[2])

        y_note = np.load(track_dir / 'y_note.npy')
        frame_level = eval_frame_level(y_note, prediction['note'], cfg.model.decoding.frame_threshold)
        all_measures[test_id] = {f'MPE_{k}': v for k, v in frame_level.items()}

        predicted_note_events = output_to_note_events(prediction, times, **cfg.model.decoding)
        if len(predicted_note_events) == 0:
            print(f"Warning: no note events predicted for {test_id}")
        pending_note_evals[test_id] = (predicted_note_events, track_dir, hcqt_cfg, cfg.data.annotations.min_note)

        note_events_to_midi(predicted_note_events).write(str(predictions_dir / f'{test_id}.mid'))
        if cfg.save_frame_predictions:
            np.save(predictions_dir / f'{test_id}.npy', prediction['note'])
            np.save(predictions_dir / f'{test_id}_onset.npy', prediction['onset'])

    with Pool(processes=min(10, os.cpu_count())) as pool:
        results = list(tqdm(pool.imap(evaluate_note_events, pending_note_evals.values()),
                            total=len(pending_note_evals), desc="Note-level eval"))
    for test_id, measures in zip(pending_note_evals.keys(), results):
        all_measures[test_id].update(measures)

    # track-wise measures, e.g. for evaluation per dataset
    df = pd.DataFrame.from_dict(all_measures, orient='index')
    df.index.name = 'id'
    log_dir = Path(cfg.run_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_dir / 'track_wise_evaluation.csv')

    mean_measures = df.mean().to_dict()
    print('Mean measures over all files:')
    for key, value in mean_measures.items():
        print(f'{key:30}: {value:.6f}')
    return mean_measures
