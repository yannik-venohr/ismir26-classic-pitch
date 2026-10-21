"""
Evaluate the MIDI output of all models in Table 5 of the paper on the test set:
the pre-trained baselines (MT3, NoteEM, BP) and our runs (outputs/<run>/predictions/).

Expects one {id}.mid per test track and writes the track-wise measures to results/note_level/<model>.csv.
The ground truth is read from the precomputed features (computed by train.py).
"""
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from src.data.data_module import feature_dir_for_config
from src.data.utils import midi_file_to_note_events_with_pedal
from src.eval.evaluate import evaluate_note_events

ENVIRONMENT = "default"                     # configs/environment/<ENVIRONMENT>.yaml, for the feature dir
RUNS_DIR = Path("outputs")                  # our runs: RUNS_DIR/<model>/predictions/
BASELINES_DIR = Path("baseline_predictions")  # pre-trained baselines: BASELINES_DIR/<model>/predictions/
OUT_DIR = Path("results/note_level")

BASELINES = ["BP", "MT3", "NoteEM"]
RUNS = [
    "CP-L_42",
    "CP-L_43",
    "CP-L_44",
    "CP-XL_42",
    "CP-M_42",
    "CP-S_42",
    "CP-XS_42",
    "BP+_42",
    "CP-MusicNet_42",
    "CP-Less_42",
    "L1-BCE_42",
    "L2-BCE+tolerance_42",
    "L3-BCE+weight_42",
    "CP-frame_loss_42",
    "CP-onset_loss_42",
    "R1-Overfit_42",
]


def evaluate_track(args):
    test_id, midi_path, track_dir, hcqt_cfg, min_note = args
    if not midi_path.exists():
        print(f"WARNING: no prediction for {test_id}: {midi_path}")
        return test_id, None
    predicted_note_events = midi_file_to_note_events_with_pedal(midi_path)
    return test_id, evaluate_note_events((predicted_note_events, track_dir, hcqt_cfg, min_note))


def evaluate_model(predictions_dir, model, cfg, test_ids):
    feature_dir = feature_dir_for_config(cfg.data)
    jobs = [(test_id, predictions_dir / f"{test_id}.mid", feature_dir / test_id, cfg.data.hcqt, cfg.data.annotations.min_note)
            for test_id in test_ids]
    with Pool(processes=8) as pool:
        results = dict(tqdm(pool.imap(evaluate_track, jobs), total=len(jobs), desc=model))

    df = pd.DataFrame.from_dict({k: v for k, v in results.items() if v is not None}, orient="index")
    df.index.name = "id"
    df.to_csv(OUT_DIR / f"{model}.csv")


if __name__ == "__main__":
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose("config", overrides=[f"environment={ENVIRONMENT}"])
    splits = pd.read_csv(cfg.data.splits_file)
    test_ids = splits.loc[splits["test"], "id"].tolist()
    print(f"Number of test tracks: {len(test_ids)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in RUNS:
        evaluate_model(RUNS_DIR / model / "predictions", model, cfg, test_ids)
    for model in BASELINES:
        evaluate_model(BASELINES_DIR / model / "predictions", model, cfg, test_ids)
