import pandas as pd

from niftsy.text.detect import describe_text_columns, describe_word_counts


def test_describe_word_counts_basic_stats():
    df = pd.DataFrame({"bio": ["one two", "one two three", "one two three four"]})
    stats = describe_word_counts(df, "bio")
    assert stats["count"] == 3
    assert stats["mean"] == 3.0
    assert stats["max"] == 4


def test_describe_word_counts_empty_column():
    df = pd.DataFrame({"bio": ["", None, "   "]})
    stats = describe_word_counts(df, "bio")
    assert stats == {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}


def test_describe_text_columns_multiple():
    df = pd.DataFrame({"bio": ["one two"], "notes": ["a b c d"]})
    stats = describe_text_columns(df, ["bio", "notes"])
    assert set(stats.keys()) == {"bio", "notes"}
    assert stats["bio"]["mean"] == 2.0
    assert stats["notes"]["mean"] == 4.0
