"""T1 · src/spatial_vocab.py — single source of truth for spatial keyword logic.

Addresses review findings C1, C2: L3 bare-substring matching + duplicate keyword lists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spatial_vocab import (  # noqa: E402
    L2_WORDS,
    L3_PHRASES,
    L3_WORDS,
    PIVOT_PATTERNS,
    classify_level,
    count_pivot_patterns,
    count_spatial_keywords,
    match_l2,
    match_l3,
)


# ---------------- classify_level ----------------


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How many apples are on the table?", "L1"),
        ("What color is the cat?", "L1"),
        ("Is the dog to the left of the cat?", "L2"),
        ("Which object is closer to the camera?", "L2"),
        ("What is directly above the lamp?", "L2"),
        ("What is the distance between the two chairs?", "L2"),
        (
            "From the camera's perspective, which object would you reach first?",
            "L3",
        ),
        (
            "If you walked forward from this viewpoint, what would you encounter?",
            "L3",
        ),
        ("Navigate to the red chair; what blocks your path?", "L3"),
        ("Reach the mug on the left", "L3"),
    ],
)
def test_classify_level_happy(question, expected):
    assert classify_level(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # These MUST NOT be classified as L3 — they contain L3 keywords as substrings only.
        "I see a sidewalk in the image.",       # "walk" in sidewalk
        "What research does this poster show?",  # "reach" fallback only if plain; here "research"
        "Is this location unapproachable?",      # "approach" in unapproachable
        "What is the teacher doing?",            # "reach" appears inside? no — "teach"
    ],
)
def test_classify_level_l3_substring_false_positives(question):
    # After fix, bare-substring matching must NOT fire on sidewalk / unapproachable.
    assert classify_level(question) != "L3", f"wrongly classified L3: {question}"


def test_classify_level_l3_beats_l2_when_both_hit():
    # "reach" (L3 word) + "left" (L2) → L3
    assert classify_level("Reach for the object to the left of the table") == "L3"


def test_classify_level_empty_is_l1():
    assert classify_level("") == "L1"
    assert classify_level(None) == "L1"


def test_classify_level_case_insensitive():
    assert classify_level("REACH the mug") == "L3"
    assert classify_level("IS A TO THE LEFT OF B") == "L2"


# ---------------- match_l2 / match_l3 ----------------


def test_match_l3_phrase():
    assert match_l3("from the camera's angle")
    assert match_l3("from your perspective this looks odd")


def test_match_l3_word_boundary():
    assert match_l3("please reach for it")
    assert not match_l3("the research budget")
    assert not match_l3("sidewalk")
    assert not match_l3("unapproachable")


def test_match_l2_word_boundary():
    assert match_l2("is the cat behind the dog?")
    assert not match_l2("behindsight is 20/20")
    assert match_l2("it is farther than the other")
    # "far" alone must not match when it's inside "farther" (but "farther" is its own keyword)
    assert match_l2("far from here")


# ---------------- counting ----------------


def test_count_spatial_keywords_basic():
    assert count_spatial_keywords("left and right") == 2
    assert count_spatial_keywords("LEFT and Right") == 2  # case-insensitive


def test_count_spatial_keywords_word_boundary():
    assert count_spatial_keywords("behindsight foreground") == 0
    assert count_spatial_keywords("the cat is behind the dog") == 1


def test_count_spatial_keywords_empty():
    assert count_spatial_keywords("") == 0
    assert count_spatial_keywords(None) == 0


def test_count_pivot_patterns_basic():
    assert count_pivot_patterns("I think, therefore I am") == 1
    assert count_pivot_patterns("A is first, then B") >= 1


def test_count_pivot_patterns_zero():
    assert count_pivot_patterns("no pivots here at all") == 0


def test_count_pivot_patterns_case_insensitive():
    assert count_pivot_patterns("THEREFORE we can conclude") >= 1


# ---------------- vocab integrity ----------------


def test_vocab_tuples_are_frozen():
    assert isinstance(L2_WORDS, tuple)
    assert isinstance(L3_WORDS, tuple)
    assert isinstance(L3_PHRASES, tuple)
    assert isinstance(PIVOT_PATTERNS, tuple)


def test_vocab_no_overlap_between_l2_and_l3_words():
    # Design invariant: a word is either L2 or L3, not both.
    assert set(L2_WORDS).isdisjoint(set(L3_WORDS))


def test_vocab_all_lowercase():
    for kw in L2_WORDS + L3_WORDS:
        assert kw == kw.lower(), f"keyword not lowercase: {kw}"
    for phrase in L3_PHRASES:
        assert phrase == phrase.lower(), f"phrase not lowercase: {phrase}"
