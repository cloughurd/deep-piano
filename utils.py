import numpy as np
import pretty_midi
import librosa

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

lowest_key = 21
highest_key = 108
octave_size = 12
desired_sr = 22050
window_size = 7
pretty_midi.pretty_midi.MAX_TICK = 1e10

def wav_to_input(fn, bin_multiple=3):
  bins_per_octave = bin_multiple * octave_size
  num_bins = (highest_key+1 - lowest_key) * bin_multiple
  
  audio, _ = librosa.load(fn, desired_sr)
  print(audio.shape)
  cqt = librosa.cqt(audio, desired_sr, fmin=librosa.midi_to_hz(lowest_key), bins_per_octave=bins_per_octave, n_bins=num_bins)
  del audio
  cqt = cqt.T # Puts time dim first
  cqt = np.abs(cqt)
  min_fq = np.min(cqt)
  cqt = np.pad(cqt, ((window_size//2, window_size//2),(0,0)), 'constant', constant_values=min_fq)

  # This sets up a matrix where at each time step we have a 7 (window_size) frame snippet from which to pull piano pitches
  windows = []
  for i in range(len(cqt) - window_size + 1):
    windows.append(cqt[i:i+window_size, :])
  cqt = np.array(windows)
  print(cqt.shape)
  return cqt

def midi_to_output(midi, x):
  times = librosa.frames_to_time(np.arange(len(x)), desired_sr)
  roll = midi.get_piano_roll(desired_sr, times)
  roll = roll[lowest_key: highest_key+1]
  roll = roll.T # Puts time dim first
  roll[roll > 0] = 1
  return roll