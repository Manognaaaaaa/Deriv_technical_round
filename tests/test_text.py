"""Deterministic text preprocessing."""

from text import content_terms, numbers, query_terms, split_sentences, stem, tokenize


def test_stemmer_examples():
    assert stem("address") == "address"
    assert stem("addresses") == "address"
    assert stem("reviews") == "review"
    assert stem("supported") == "support"
    assert stem("loses") == "lose"
    assert stem("policies") == "policy"
    assert stem("closing") == "clos"
    assert stem("boxes") == "box"


def test_stemmer_keeps_three_chars_and_digits():
    assert stem("sing") == "sing"  # "ing" would leave 1 char
    assert stem("24") == "24"
    assert stem("2000s") == "2000"  # not all digits, so "s" is stripped
    assert stem("class") == "class"  # never strip "s" after "ss"


def test_tokenize_keeps_2fa_as_one_token_and_digits():
    assert tokenize("What 2FA methods?") == ["what", "2fa", "methods"]
    assert tokenize("within 24 hours") == ["within", "24", "hours"]


def test_tokenize_nfkc():
    assert tokenize("ｒｅｓｅｔ") == ["reset"]  # full-width letters normalised


def test_filler_removed_from_questions_only():
    assert query_terms("What do I need before closing my account?") == ["before", "clos", "account"]
    assert "need" in content_terms("You need a code")


def test_numbers_are_whole_digit_tokens():
    assert numbers("Enable 2FA within 24 hours") == {"24"}


def test_sentence_offsets_slice_original_text():
    text = "First one.  Second one! Third? Last without end"
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["First one.", "Second one!", "Third?", "Last without end"]
