import pytest
from phase1_normalization import (
    sanitize_unicode,
    flatten_whitespace,
    convert_words_to_digits,
    run_phase1_normalization
)

def test_unicode_sanitization():
    # Tests stripping of U+200B (Zero-width space)
    raw = "hello\u200Bworld"
    assert sanitize_unicode(raw) == "helloworld"

def test_whitespace_flattening():
    # Tests flattening of \n, \t, and multi-spaces
    raw = "line1\n\nline2\t  space"
    assert flatten_whitespace(raw) == "line1 line2 space"

def test_word_to_digit():
    raw = "my number is n1ne-eight-s3ven-65-43-21-0"
    assert convert_words_to_digits(raw) == "my number is 9-8-7-65-43-21-0"

def test_word_to_digit2():
    raw = "call me at 9️⃣8️⃣7️⃣6️⃣5️⃣4️⃣3️⃣2️⃣1️⃣0️⃣ once you land"
    assert run_phase1_normalization(raw)["final_text"] == "call me at 9876543210 once you land"

def test_email_verbal_detection_and_masking_copy1():
    raw = "Contact me at john at example dot com urgently"
    result = run_phase1_normalization(raw)
    print("copy1:", result)
    assert result["final_text"] == "Contact me at [REDACTED_EMAIL] urgently"
    assert len(result["spans_detected"]) == 1


def test_email_verbal_detection_and_masking_copy2():
    raw = "send your ID scan to priya dot host at gmail dot com"
    result = run_phase1_normalization(raw)
    print("copy2:", result)
    assert result["final_text"] == "send your ID scan to [REDACTED_EMAIL]"
    assert len(result["spans_detected"]) == 1


def test_email_verbal_detection_and_masking_copy3():
    raw = "email kar do details, priya.stays@gmail dawt com pe"
    result = run_phase1_normalization(raw)
    print("copy3:", result)
    assert result["final_text"] == "email kar do details, [REDACTED_EMAIL] pe"
    assert len(result["spans_detected"]) == 1


def test_email_verbal_detection_and_masking_copy4():
    raw = "reach me a t alex [at] gmail [dot] com for the deposit"
    result = run_phase1_normalization(raw)
    print("copy4:", result)
    assert result["final_text"] == "reach me at [REDACTED_EMAIL] for the deposit"
    assert len(result["spans_detected"]) == 1


def test_email_verbal_detection_and_masking_copy5():
    raw = "email works better - r a h u l 1 9 9 2 @ y a h o o . c o m"
    result = run_phase1_normalization(raw)
    print("copy5:", result)
    assert result["final_text"] == "email works better - [REDACTED_EMAIL]"
    assert len(result["spans_detected"]) == 1

def test_email_verbal_detection_and_masking_copy6():
    raw = "email me the invoice - j.dsouza1988@hotmail dot com"
    result = run_phase1_normalization(raw)
    print("copy6:", result)
    assert result["final_text"] == "email me the invoice - [REDACTED_EMAIL]"
    assert len(result["spans_detected"]) == 1

def test_email_verbal_detection_and_masking_copy7():
    raw = "my mail is s.narayan.stays@rediffmail.com, send the docs there"
    result = run_phase1_normalization(raw)
    print("copy7:", result)
    assert result["final_text"] == "my mail is [REDACTED_EMAIL], send the docs there"
    assert len(result["spans_detected"]) == 1

def test_url_detection():
    # Standard URL extraction
    raw = "quicker on telegram - t.me / villa_owner_goa"
    result = run_phase1_normalization(raw)
    
    assert result["final_text"] == "quicker on telegram - [REDACTED_URL]"
    assert len(result["spans_detected"]) == 1

def test_no_match_forwarding_and_revert():
    # Tests the "no confirmed email/URL" branch in the UML
    raw = "Meet me at the park at nine"
    result = run_phase1_normalization(raw)
    
    # Canonicalization turns it into "Meet me@the park@9"
    # But because it fails regex detection, it hits the revert block -> "Meet me at the park at 9"
    assert result["final_text"] == "Meet me at the park at 9"
    assert len(result["spans_detected"]) == 0


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("n1ne", "9"),
        ("s3ven", "7"),
        ("zer0", "0"),
        ("Call at n1ne s3ven", "Call at 9 7"),
        ("one-two-three", "1-2-3"),
    ],
)
def test_word_to_digit_obfuscated_and_delimited_variants(raw, expected):
    assert convert_words_to_digits(raw) == expected


def test_word_to_digit_does_not_convert_unrelated_words():
    raw = "stone tone sevenly"
    assert convert_words_to_digits(raw) == "stone tone sevenly"


def test_word_to_digit_keeps_plain_numbers_unchanged():
    raw = "pin is 9876543210"
    assert convert_words_to_digits(raw) == "pin is 9876543210"


def test_phase1_masks_www_url_variant():
    raw = "book direct on www.goavillarentals.com today"
    result = run_phase1_normalization(raw)
    print("www_url:", result)
    assert result["final_text"] == "book direct on [REDACTED_URL] today"
    assert len(result["spans_detected"]) == 1


def test_phase1_masks_multiple_contacts_in_one_message():
    raw = "email me at a.b@example.com and check test.com"
    result = run_phase1_normalization(raw)
    print("multi_contact:", result)
    assert result["final_text"] == "email me at [REDACTED_EMAIL] and check [REDACTED_URL]"
    assert len(result["spans_detected"]) == 2


def test_phase1_obfuscated_number_no_contact_still_forwards():
    raw = "we are n1ne people arriving tomorrow"
    result = run_phase1_normalization(raw)
    print("obfuscated_non_contact:", result)
    assert result["final_text"] == "we are 9 people arriving tomorrow"
    assert len(result["spans_detected"]) == 0