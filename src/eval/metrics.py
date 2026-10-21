import numpy as np
import mir_eval
import librosa
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from src.data.utils import filter_duplicate_onsets_by_frame, filter_invalid_note_events


def eval_frame_level(target, prediction, threshold=0.5):
    """Frame-level evaluation measures for multi-pitch estimation.

    Args:
        target:     Binary array of target, size (n_bins, n_frames)
        prediction: Prediction array (probabilities between zero and one of same size)
        threshold:  Threshold for binarizing the prediction (precision, recall, f_meas)

    Returns:
        Dictionary containing names and values of the evaluation measures.
    """
    target = target.flatten()
    prediction = prediction.flatten()
    prediction_thresh = prediction >= threshold
    return {
        'precision': precision_score(target, prediction_thresh),
        'recall': recall_score(target, prediction_thresh),
        'f_meas': f1_score(target, prediction_thresh),
        'average_precision': average_precision_score(target, prediction),
    }


def eval_note_level(note_events_target, note_events_prediction, tolerances=[0.05, 0.1, 0.2], frame_size=512 / 22050):
    """Note-level evaluation measures using mir_eval, for several onset tolerances.

    Args:
        note_events_target:     List of target note events (start, end, midi pitch)
        note_events_prediction: List of predicted note events (start, end, midi pitch, amplitude)
        tolerances:             Onset tolerances in seconds (also used as minimum offset tolerance)
        frame_size:             Frame size in seconds, notes of the same pitch with onsets in the same frame are duplicates
    Returns:
        Dictionary with keys like '50_F-measure_no_offset' (tolerance in ms + mir_eval measure).
    """
    # filter out duplicate onsets (relevant for multi-track datasets such as PHENICX)
    n_before = len(note_events_target)
    note_events_target = filter_invalid_note_events(filter_duplicate_onsets_by_frame(note_events_target, frame_size))
    if len(note_events_target) < n_before:
        print(f"Warning: {n_before - len(note_events_target)} duplicate or invalid target note events were filtered out.")

    ref_intervals = np.array([[start, end] for start, end, *_ in note_events_target])
    ref_pitches = librosa.midi_to_hz(np.array([pitch for _, _, pitch, *_ in note_events_target]))

    note_events_prediction = filter_duplicate_onsets_by_frame(note_events_prediction, frame_size)
    if len(note_events_prediction) == 0:
        est_intervals = np.zeros((0, 2))
        est_pitches = np.zeros(0)
    else:
        est_intervals = np.array([[start, end] for start, end, *_ in note_events_prediction])
        est_pitches = librosa.midi_to_hz(np.array([pitch for _, _, pitch, *_ in note_events_prediction]))

    measures_dict_all = {}
    for tolerance in tolerances:
        measures_dict = mir_eval.transcription.evaluate(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=tolerance,
            offset_min_tolerance=tolerance,
        )
        measures_dict_all.update({f'{int(tolerance*1000)}_{key}': value for key, value in measures_dict.items()})

    return measures_dict_all
