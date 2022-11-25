import note_seq
import numpy as np
import torch
from typing import Any, Callable, MutableMapping, Optional, Sequence, Tuple, TypeVar

from .event_codec import (
    Codec,
    Event,
    NoteEventData,
    NoteEncodingState,
    NoteDecodingState,
)


CODEC = Codec(512)


def note_encoding_state_to_events(
    state: NoteEncodingState, codec: Codec
) -> Sequence[Event]:
    """Output program and pitch events for active notes plus a final tie event."""
    events = []
    offset_value = codec.encode_event(Event(type="velocity", value=0))
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: k[::-1]):
        if state.active_pitches[(pitch, program)] != offset_value:
            # events += [Event("program", program), Event("pitch", pitch)]
            events += [program, pitch]
    events.append(codec.encode_event(Event(type="tie", value=0)))
    return events


def read_midi(filename):
    with open(filename, "rb") as f:
        content = f.read()
        ns = note_seq.midi_to_note_sequence(content)
    return ns


def tokenize(ns: note_seq.NoteSequence, frame_rate: int, codec: Codec):
    notes = sorted(ns.notes, key=lambda note: (
        note.is_drum, note.program, note.pitch))
    # times = [note.end_time*frame_rate for note in notes if not note.is_drum] + [
    #     note.start_time*frame_rate for note in notes
    # ]
    offset_times = np.round(
        [note.end_time * frame_rate for note in notes if not note.is_drum]
    )
    onset_times = np.round([note.start_time * frame_rate for note in notes])
    times = np.concatenate((offset_times, onset_times), axis=0).astype(int)

    values = [
        codec.encode_note(note, velocity=0) for note in notes if not note.is_drum
    ] + [codec.encode_note(note) for note in notes]
    return times, values


def preprocess(ns, resolution=100, segment_length=5.12, output_size=2048, codec=CODEC):
    """Preprocess MIDI tokens to pytorch Tensors of integers with output size 
    and segemented into chucks of segement length. 
    Args:
      ns: note sequence of MIDI notes
      resolution: number of frames/steps per second
      segment_length: number of seconds per segment
      output_size: length of output tensor of each segment
      codec: vocabulary and encoding object of class Codec
    Returns:
      (segmented tokens: number_of_segment x output_size,
       segment times: number_of_segment x 2)
    """
    segment_length = np.ceil(segment_length * resolution).astype(int)
    steps, values = tokenize(ns, resolution, codec)
    stamps = np.unique(steps)
    num_segments = np.ceil(stamps[-1] / segment_length).astype(int)
    events = {}
    state_events = {0: [codec.encode_event(Event(type="tie", value=0))]}
    ds = NoteEncodingState()
    segments, shifts = np.divmod(stamps, segment_length)
    change_points = np.zeros_like(segments)
    change_points[:-1] = segments[1:] != segments[:-1]
    for i, stamp in enumerate(stamps):
        segment_num, shift_num = segments[i], shifts[i]
        event_idx = np.nonzero(steps == stamp)[0]
        event_values = [values[i] for i in event_idx]
        event = events.get(segment_num, [])
        event = event + ([shift_num] * (shift_num > 0)) + \
            [v for e in event_values for v in e]
        events[segment_num] = event
        for value in event_values:
            if len(value) == 3:
                ds.active_pitches[(value[-1], value[0])] = value[1]
        if change_points[i]:
            state_event = note_encoding_state_to_events(ds, codec)
            state_events[segment_num + 1] = state_event
    tokens = torch.zeros(num_segments, output_size, dtype=torch.long)
    for k, v in events.items():
        all_events = torch.Tensor(state_events.get(k, []) + v)
        tokens[k, :len(all_events)] = all_events
    segment_time = segment_length / resolution
    segment_times = [
        (i * segment_time, (i + 1) * segment_time) for i in range(num_segments - 1)
    ]
    segment_times.append(((num_segments - 1) * segment_time, ns.total_time))
    return tokens, torch.Tensor(segment_times)


############################################################################
#                       Decoding Functions                                 #
############################################################################


def decode_events(
    state: Any,
    tokens: np.ndarray,
    start_time: int,
    max_time: Optional[int],
    codec: Codec,
    resolution: int,
    decode_event_fn: Callable[[Any, float, Event, Codec], None],
) -> Tuple[int, int]:
    """Decode a series of tokens, maintaining a decoding state object.

    Args:
      state: Decoding state object; will be modified in-place.
      tokens: event tokens to convert.
      start_time: offset start time if decoding in the middle of a sequence.
      max_time: Events at or beyond this time will be dropped.
      codec: An event_codec.Codec object that maps indices to Event objects.
      decode_event_fn: Function that consumes an Event (and the current time) and
          updates the decoding state.

    Returns:
      invalid_events: number of events that could not be decoded.
      dropped_events: number of events dropped due to max_time restriction.
    """
    invalid_events = 0
    dropped_events = 0
    cur_steps = 0
    cur_time = start_time
    token_idx = 0
    for token_idx, token in enumerate(tokens):
        try:
            event = codec.decode_event_index(int(token.item()))
        except ValueError:
            invalid_events += 1
            continue
        if event.type == "shift":
            cur_steps += event.value
            cur_time = start_time + cur_steps / resolution
            if max_time and cur_time > max_time:
                dropped_events = len(tokens) - token_idx
                break
        else:
            cur_steps = 0
            try:
                decode_event_fn(state, cur_time, event)
            except ValueError as e:
                invalid_events += 1
                continue
    return invalid_events, dropped_events


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: Event,
) -> None:
    """Process note event and update decoding state."""
    if time < state.current_time:
        raise ValueError(
            "event time < current time, %f < %f" % (time, state.current_time)
        )
    state.current_time = time
    if event.type == "pitch":
        pitch = event.value
        if state.is_tie_section:
            # "tied" pitch
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(
                    "inactive pitch/program in tie section: %d/%d"
                    % (pitch, state.current_program)
                )
            if (pitch, state.current_program) in state.tied_pitches:
                raise ValueError(
                    "pitch/program is already tied: %d/%d"
                    % (pitch, state.current_program)
                )
            state.tied_pitches.add((pitch, state.current_program))
        elif state.current_velocity == 0:
            # note offset
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(
                    "note-off for inactive pitch/program: %d/%d"
                    % (pitch, state.current_program)
                )
            onset_time, onset_velocity = state.active_pitches.pop(
                (pitch, state.current_program)
            )
            _add_note_to_sequence(
                state.note_sequence,
                start_time=onset_time,
                end_time=time,
                pitch=pitch,
                velocity=onset_velocity,
                program=state.current_program,
            )
        else:
            # note onset
            if (pitch, state.current_program) in state.active_pitches:
                # The pitch is already active; this shouldn't really happen but we'll
                # try to handle it gracefully by ending the previous note and starting a
                # new one.
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, state.current_program)
                )
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time=onset_time,
                    end_time=time,
                    pitch=pitch,
                    velocity=onset_velocity,
                    program=state.current_program,
                )
            state.active_pitches[(pitch, state.current_program)] = (
                time,
                state.current_velocity,
            )
    elif event.type == "drum":
        # drum onset (drums have no offset)
        if state.current_velocity == 0:
            raise ValueError("velocity cannot be zero for drum event")
        offset_time = time + 0.01
        _add_note_to_sequence(
            state.note_sequence,
            start_time=time,
            end_time=offset_time,
            pitch=event.value,
            velocity=state.current_velocity,
            is_drum=True,
        )
    elif event.type == "velocity":
        # velocity change
        state.current_velocity = event.value
    elif event.type == "program":
        # program change
        state.current_program = event.value
    elif event.type == "tie":
        # end of tie section; end active notes that weren't declared tied
        if not state.is_tie_section:
            raise ValueError("tie section end event when not in tie section")
        for (pitch, program) in list(state.active_pitches.keys()):
            if (pitch, program) not in state.tied_pitches:
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, program))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time=onset_time,
                    end_time=state.current_time,
                    pitch=pitch,
                    velocity=onset_velocity,
                    program=program,
                )
        state.is_tie_section = False
    else:
        raise ValueError("unexpected event type: %s" % event.type)


def assign_instruments(ns: note_seq.NoteSequence) -> None:
    """Assign instrument numbers to notes; modifies NoteSequence in place."""
    program_instruments = {}
    for note in ns.notes:
        if note.program not in program_instruments and not note.is_drum:
            num_instruments = len(program_instruments)
            note.instrument = (
                num_instruments if num_instruments < 9 else num_instruments + 1
            )
            program_instruments[note.program] = note.instrument
        elif note.is_drum:
            note.instrument = 9
        else:
            note.instrument = program_instruments[note.program]


def flush_note_decoding_state(state: NoteDecodingState) -> note_seq.NoteSequence:
    """End all active notes and return resulting NoteSequence."""
    for onset_time, _ in state.active_pitches.values():
        state.current_time = max(state.current_time, onset_time + 0.01)
    for (pitch, program) in list(state.active_pitches.keys()):
        onset_time, onset_velocity = state.active_pitches.pop((pitch, program))
        _add_note_to_sequence(
            state.note_sequence,
            start_time=onset_time,
            end_time=state.current_time,
            pitch=pitch,
            velocity=onset_velocity,
            program=program,
        )
    assign_instruments(state.note_sequence)
    return state.note_sequence


def tokens_to_notes(tokens, codec):
    decoding_state = NoteDecodingState()
    decoding_state.is_tie_section = True
    decoding_state.current_velocity = 0
    decode_events(
        state=decoding_state,
        tokens=tokens,
        start_time=0,
        max_time=None,
        codec=codec,
        resolution=100,
        decode_event_fn=decode_note_event,
    )
    return flush_note_decoding_state(decoding_state)


def _add_note_to_sequence(
    ns: note_seq.NoteSequence,
    start_time: float,
    end_time: float,
    pitch: int,
    velocity: int,
    program: int = 0,
    is_drum: bool = False,
) -> None:
    end_time = max(end_time, start_time + 0.01)
    ns.notes.add(
        start_time=start_time,
        end_time=end_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum,
    )
    ns.total_time = max(ns.total_time, end_time)
