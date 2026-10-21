import numpy as np
import librosa
import pretty_midi
from pathlib import Path


############# HCQT ############
def audio_to_hcqt(audio, sr_audio, param_hcqt):
    """
        Compute Harmonic CQT

        Parameters
        ----------
        audio : np.array
            mono audio signal as np.array
        sr_audio : int
            sampling rate of the audio signal
        param_hcqt : dict
            configuration parameters for the Harmonic CQT

        Returns
        -------
        data_3m : np.array
            3D array of Harmonic CQT
        time_sec_v : np.array
            corresponding time [in sec] of analysis windows
        frequency_hz_v : np.array
            corresponding frequency [in Hz] of 1st harmonic bin
    """
    fmin = float(librosa.midi_to_hz(param_hcqt.min_note))
    tuning_est = librosa.estimate_tuning(y=audio)
    fmin_tuned = fmin * 2**(tuning_est / param_hcqt.bins_per_octave)

    hcqt_list = []
    min_time_frames = float('inf')
    for h in param_hcqt.harmonics:
        A_m = np.abs(librosa.cqt(y=audio, sr=sr_audio,
                                 fmin=h*fmin_tuned,
                                 hop_length=param_hcqt.hop_length,
                                 bins_per_octave=param_hcqt.bins_per_octave,
                                 n_bins=param_hcqt.n_bins,
                                 tuning=0.0,
                                 ))
        hcqt_list.append(A_m)
        min_time_frames = min(min_time_frames, A_m.shape[1])

    # Trim all HCQT arrays to the smallest frame length
    hcqt_list = [A[:, :min_time_frames] for A in hcqt_list]
    data_3m = np.stack(hcqt_list, axis=0)

    n_times = data_3m.shape[2]
    time_sec_v = librosa.frames_to_time(np.arange(n_times),
                                        sr=sr_audio,
                                        hop_length=param_hcqt.hop_length)

    frequency_hz_v = librosa.cqt_frequencies(n_bins=param_hcqt.n_bins,
                                             fmin=fmin_tuned,
                                             bins_per_octave=param_hcqt.bins_per_octave)

    return data_3m, time_sec_v, frequency_hz_v


def get_hcqt_times(param_hcqt, n_frames):
    """Get time stamps [in sec] for HCQT frames."""
    return librosa.frames_to_time(np.arange(n_frames),
                                  sr=param_hcqt.sr,
                                  hop_length=param_hcqt.hop_length)


########### MIDI AND NOTE EVENTS ############
def note_events_to_midi(note_events, output_file=None, instrument_name='Acoustic Grand Piano'):
    """
    Convert a list of note events to a MIDI file using PrettyMIDI.

    Parameters:
        note_events (list of tuples): Each tuple contains
            (start_time, end_time, midi_number) or
            (start_time, end_time, midi_number, amplitude, ...),
            where amplitude is a float from 0.0 to 1.0
            (defaults to 1.0 if not given).
        output_file (str): Path to save the generated MIDI file.
        instrument_name (str): Name of the instrument for the MIDI track.
    """
    midi = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    for start, end, pitch, *rest in note_events:
        amplitude = rest[0] if rest else 1.0
        velocity = int(np.clip(amplitude * 127, 0, 127))
        note = pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=start, end=end)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    if output_file is not None:
        midi.write(output_file)
    return midi


def midi_file_to_note_events(midi):
    """Convert a midi file to a list of note events. Drum tracks are ignored.

    Args:
        midi: Path to a midi file

    Returns:
        A list of note events [start, end, pitch, amplitude]
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi))

    note_events = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            note_events.append([note.start, note.end, note.pitch, note.velocity / 127.0])
    return note_events


def midi_file_to_note_events_with_pedal(midi_file):
    """Convert a midi file to a list of note events, extending notes that are held by the sustain pedal.

    Args:
        midi_file: Path to a midi file

    Returns:
        A list of note events [start, end, pitch]
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))

    note_events = []
    for instrument in midi_data.instruments:
        sustain_times = []  # (press, release) times of the sustain pedal
        is_pedal_on = False
        for control in instrument.control_changes:
            if control.number == 64:  # Sustain pedal (CC 64)
                is_current_pedal_on = (control.value >= 64)
                if not is_pedal_on and is_current_pedal_on:
                    time_pedal_on = control.time
                    is_pedal_on = True
                elif is_pedal_on and not is_current_pedal_on:
                    sustain_times.append((time_pedal_on, control.time))
                    is_pedal_on = False

        for note in instrument.notes:
            start, end, pitch = note.start, note.end, note.pitch
            # Adjust end time based on sustain pedal activity
            for pedal_start, pedal_end in sustain_times:
                if start < pedal_end and end <= pedal_start:
                    end = pedal_end
                    break
            note_events.append([start, end, pitch])

    return note_events


def midi_files_to_note_events(midi_dir: Path):
    """Collect the note events of all (non-percussion) midi files in a directory."""
    all_note_events = []
    for midi in midi_dir.glob('*.mid'):
        if 'kick' in midi.name or 'snare' in midi.name or 'ride' in midi.name:
            continue
        all_note_events.extend(midi_file_to_note_events(midi))
    return all_note_events


######### NOTE EVENT PROCESSING #########
def filter_duplicate_onsets_by_frame(note_events, frame_size):
    """Filter duplicate note events considering a given frame size.
    Two notes with the same pitch are considered duplicates if their onsets
    fall within the same frame. The longer duration note is kept.

    Args:
        note_events: A list of note events where each note is specified as
                     [start, end, pitch] or [start, end, pitch, amplitude, ...]
        frame_size: Frame size in seconds.
    Returns:
        filtered_events: A list of note events with frame-level duplicates removed.
                         Any extra fields (e.g. amplitude) are preserved.
    """
    unique_events = {}
    for event in note_events:
        start, end, pitch = event[0], event[1], event[2]
        frame_idx = int(start // frame_size)
        key = (frame_idx, pitch)
        duration = end - start
        if key not in unique_events or duration > (unique_events[key][1] - unique_events[key][0]):
            unique_events[key] = event
    return list(unique_events.values())


def filter_invalid_note_events(note_events):
    """Remove note events whose end time is not after their start time."""
    return [event for event in note_events if event[0] < event[1]]


######### NOTE EVENTS TO ARRAYS #########
def note_events_to_binary_mpe_array(note_events, times, min_note, n_pitch_bins, add_onsets=False):
    """
    Convert a list of note events to a binary multi-pitch array.

    Parameters
    ----------
    note_events : list
        List of note events in the format (start_time, end_time, pitch (midi))
        or (start_time, end_time, pitch (midi), amplitude, ...); extra fields
        are ignored.
    times : np.array
        Array of time stamps in seconds.
    min_note : int
        Minimum note in MIDI format.
    n_pitch_bins : int
        Number of pitch bins.
    add_onsets : bool, optional
        If True, onset frames are marked with 2 and active frames with 1.

    Returns
    -------
    mpe_array : np.ndarray
        Multi-pitch array of shape (n_pitch_bins, len(times)).
    """
    pitch_bins = np.arange(min_note, min_note + n_pitch_bins)
    mpe_array = np.zeros((n_pitch_bins, len(times)))

    for start, end, pitch, *_ in note_events:
        start_idx = np.argmin(np.abs(start - times))
        end_idx = np.argmin(np.abs(end - times))
        pitch_idx = np.argmin(np.abs(pitch - pitch_bins))
        mpe_array[pitch_idx, start_idx:end_idx] = 1
        if add_onsets:
            mpe_array[pitch_idx, start_idx] = 2

    return mpe_array


def note_events_to_onset_array(note_events, times, min_note, n_pitch_bins):
    """
    Convert a list of note events to a binary onset array.

    Parameters
    ----------
    note_events : list
        List of note events in the format (start_time, end_time, pitch (midi))
        or (start_time, end_time, pitch (midi), amplitude, ...); extra fields
        are ignored.
    times : np.array
        Array of time stamps in seconds.
    min_note : int
        Minimum note in MIDI format.
    n_pitch_bins : int
        Number of pitch bins.

    Returns
    -------
    onset_array : np.ndarray
        Onset array of shape (n_pitch_bins, len(times)).
    """
    pitch_bins = np.arange(min_note, min_note + n_pitch_bins)
    onset_array = np.zeros((n_pitch_bins, len(times)))

    for start, end, pitch, *_ in note_events:
        start_idx = np.argmin(np.abs(start - times))
        pitch_idx = np.argmin(np.abs(pitch - pitch_bins))
        onset_array[pitch_idx, start_idx] = 1.0

    return onset_array
