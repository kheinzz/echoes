from echoes.merge import (
    UNKNOWN_SPEAKER,
    SpeakerIndex,
    assign_speakers,
    group_consecutive,
    rename_speakers,
    split_sentences,
)
from echoes.schema import Segment, Turn, Utterance, Word


def segment(*words: tuple[float, float, str]) -> Segment:
    items = [Word(start, end, text) for start, end, text in words]
    return Segment(items[0].start, items[-1].end, "".join(w.text for w in items).strip(), items)


def texts(utterances):
    return [(u.speaker, u.text) for u in utterances]


def test_sentences_end_on_punctuation_across_segments():
    segments = [
        segment((0.0, 0.5, " Hello"), (0.5, 1.0, " there."), (1.0, 1.5, " How")),
        segment((1.5, 2.0, " are"), (2.0, 2.5, " you?")),
    ]
    sentences = ["".join(w.text for w in s).strip() for s in split_sentences(segments)]
    assert sentences == ["Hello there.", "How are you?"]


def test_standalone_punctuation_stays_with_its_sentence():
    segments = [segment((0.0, 0.5, " «Oui."), (0.5, 0.8, " »"), (0.8, 1.0, " Alors"))]
    sentences = ["".join(w.text for w in s).strip() for s in split_sentences(segments)]
    assert sentences == ["«Oui. »", "Alors"]


def test_sentences_end_on_long_pause():
    segments = [segment((0.0, 0.5, " Alors")), segment((3.0, 3.5, " voilà"))]
    sentences = ["".join(w.text for w in s).strip() for s in split_sentences(segments)]
    assert sentences == ["Alors", "voilà"]


def test_segment_without_words_is_one_unit():
    legacy = [Segment(0.0, 2.0, "Old format."), Segment(2.0, 4.0, "Still works.")]
    turns = [Turn(0.0, 2.1, "A"), Turn(2.1, 4.0, "B")]
    assert texts(assign_speakers(legacy, turns)) == [("A", "Old format."), ("B", "Still works.")]


def test_speaker_change_inside_a_whisper_segment():
    segments = [
        segment(
            (0.0, 0.4, " Are"), (0.4, 0.8, " you"), (0.8, 1.2, " ready?"),
            (1.3, 1.6, " Yes,"), (1.6, 2.0, " sure."),
        )
    ]
    turns = [Turn(0.0, 1.25, "A"), Turn(1.25, 2.0, "B")]
    assert texts(assign_speakers(segments, turns)) == [("A", "Are you ready?"), ("B", "Yes, sure.")]


def test_sentence_keeps_majority_speaker_despite_boundary_jitter():
    segments = [segment((0.0, 1.0, " This"), (1.0, 2.0, " is"), (2.0, 3.0, " mine."))]
    # the turn of B starts slightly too early
    turns = [Turn(0.0, 2.8, "A"), Turn(2.8, 5.0, "B")]
    assert texts(assign_speakers(segments, turns)) == [("A", "This is mine.")]


def test_nearest_turn_when_nothing_overlaps():
    index = SpeakerIndex([Turn(0.0, 1.0, "A"), Turn(5.0, 6.0, "B")])
    assert index.speaker_weights(1.5, 2.0) == {"A": 0.5}
    assert index.speaker_weights(4.0, 4.5) == {"B": 0.5}
    assert index.speaker_weights(8.0, 9.0) == {"B": 1.0}


def test_overlapping_turns_are_supported():
    index = SpeakerIndex([Turn(0.0, 10.0, "A"), Turn(2.0, 3.0, "B"), Turn(4.0, 5.0, "C")])
    assert index.speaker_weights(4.5, 6.0) == {"A": 1.5, "C": 0.5}
    # nearest turn before is A, whose end is the latest one
    index = SpeakerIndex([Turn(0.0, 10.0, "A"), Turn(2.0, 3.0, "B")])
    assert list(index.speaker_weights(11.0, 12.0)) == ["A"]


def test_zero_length_word_gets_the_surrounding_speaker():
    index = SpeakerIndex([Turn(0.0, 1.0, "A"), Turn(1.0, 2.0, "B")])
    assert list(index.speaker_weights(1.5, 1.5)) == ["B"]


def test_no_turns_gives_unknown_speaker():
    segments = [segment((0.0, 1.0, " Alone."))]
    assert texts(assign_speakers(segments, [])) == [(UNKNOWN_SPEAKER, "Alone.")]


def test_group_consecutive():
    utterances = [
        Utterance(0, 1, "A", "One."),
        Utterance(1, 2, "A", "Two."),
        Utterance(2, 3, "B", "Three."),
        Utterance(3, 4, "A", "Four."),
    ]
    assert [(u.start, u.end, u.speaker, u.text) for u in group_consecutive(utterances)] == [
        (0, 2, "A", "One. Two."),
        (2, 3, "B", "Three."),
        (3, 4, "A", "Four."),
    ]


def test_rename_speakers_by_order_of_appearance():
    utterances = [
        Utterance(0, 1, "SPEAKER_03", "a"),
        Utterance(1, 2, UNKNOWN_SPEAKER, "b"),
        Utterance(2, 3, "SPEAKER_00", "c"),
        Utterance(3, 4, "SPEAKER_03", "d"),
    ]
    assert [u.speaker for u in rename_speakers(utterances, "Locuteur")] == [
        "Locuteur 1",
        UNKNOWN_SPEAKER,
        "Locuteur 2",
        "Locuteur 1",
    ]
