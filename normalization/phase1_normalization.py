import re
import unicodedata
from typing import List, Dict, Any
from rapidfuzz import fuzz, process

# --- REGEX PATTERNS ---
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r'(https?://[^\s]+)|(www\.[^\s]+)|([a-zA-Z0-9.-]+\.(com|org|net|io|co|me|ly)(?:\s*/[\w-]+)?)', 
    re.IGNORECASE
)

# Common phonetic/slang variations for email syntax
DOT_VARIANTS = r'(?:dot|dawt|daht|d0t|\.)'
AT_VARIANTS = r'(?:at|aat|@)'


# ==========================================
# 1. CANONICALIZATION (Order Strictly Matters)
# ==========================================

def sanitize_unicode(text: str) -> str:
    """Strips zero-width characters and normalizes Unicode/homoglyphs."""
    text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
    # Normalize em-dashes and symbols to standard forms
    text = text.replace('—', ' — ').replace('–', ' - ')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

def despace_isolated_characters(text: str) -> str:
    """
    Collapses runs of single letters/digits separated by spaces.
    Example: 'r a h u l 1 9 9 2' -> 'rahul1992', 'a t' -> 'at', 'c o m' -> 'com'
    """
    # Repeatedly collapse single-character space sequences
    pattern = re.compile(r'(?<=\b[a-zA-Z0-9])\s+(?=[a-zA-Z0-9]\b)')
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = pattern.sub('', text)
    return text

def despace_path_separators(text: str) -> str:
    """Collapses spaces around slashes and dots that sit between two tokens.
    'telegram . com / villa owner' -> 'telegram.com/villaowner'
    """
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\s*\.\s*(?=[a-zA-Z0-9])', '.', text)
    return text

def flatten_whitespace(text: str) -> str:
    """Flattens tabs, line breaks, and multiple spaces into a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def convert_words_to_digits(text: str) -> str:
    """Converts clean and obfuscated number words to numeric digits."""
    word_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    # Common l33t substitutions so fuzzy matching can focus on near-word errors.
    confusable_map = str.maketrans({
        '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
        '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g'
    })

    def replace_token(token: str) -> str:
        lowered = token.lower()
        if lowered in word_map:
            return word_map[lowered]

        # Skip pure numbers and very short/long tokens to reduce false positives.
        if lowered.isdigit() or len(lowered) < 3 or len(lowered) > 7:
            return token

        normalized = lowered.translate(confusable_map)
        if normalized in word_map:
            return word_map[normalized]

        best = process.extractOne(
            normalized,
            word_map.keys(),
            scorer=fuzz.ratio,
            score_cutoff=88
        )
        if best:
            return word_map[best[0]]
        return token

    parts = re.split(r'(\W+)', text)
    return ''.join(
        replace_token(part) if re.fullmatch(r'[A-Za-z0-9]+', part) else part
        for part in parts
    )

def substitute_verbal(text: str) -> str:
    """
    Converts bracketed and phonetic verbal email separators to standard tokens.
    Handles: [at], (at), at, [dot], dawt, d0t, etc.
    """
    # 1. Handle bracketed/wrapped '@': [at], (at), {at}, <at>, etc.
    text = re.sub(rf'[\s\[\(\{{<]*{AT_VARIANTS}[\s\]\)\}}>]+', '@', text, flags=re.IGNORECASE)
    
    # 2. Handle bracketed/wrapped '.': [dot], [dawt], (dot), etc.
    text = re.sub(rf'[\s\[\(\{{<]*{DOT_VARIANTS}[\s\]\)\}}>]+', '.', text, flags=re.IGNORECASE)
    
    # 3. Clean up loose spaces around injected @ and .
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*', '.', text)
    return text

def revert_verbal(text: str) -> str:
    """Reverts un-redacted standalone '@' back to ' at '."""
    return text.replace('@', ' at ')


# ==========================================
# 2. PARALLEL STRUCTURAL DETECTION
# ==========================================

def detect_patterns(text: str) -> List[Dict[str, Any]]:
    """Runs parallel regex detection for Emails and URLs."""
    spans = []
    
    for match in EMAIL_PATTERN.finditer(text):
        spans.append({
            'type': 'EMAIL',
            'start': match.start(),
            'end': match.end(),
            'value': match.group(),
            'confidence': 'high'
        })
        
    for match in URL_PATTERN.finditer(text):
        overlap = any(s['start'] <= match.start() < s['end'] for s in spans)
        if not overlap:
            spans.append({
                'type': 'URL',
                'start': match.start(),
                'end': match.end(),
                'value': match.group(),
                'confidence': 'high'
            })
            
    return spans


# ==========================================
# 3. MASKING & RESULT ASSEMBLY
# ==========================================

def mask_matched_spans(text: str, spans: List[Dict[str, Any]]) -> str:
    """Redacts detected spans from the text."""
    sorted_spans = sorted(spans, key=lambda x: x['start'], reverse=True)
    masked_text = text
    for span in sorted_spans:
        masked_text = (
            masked_text[:span['start']] + 
            f"[REDACTED_{span['type']}]" + 
            masked_text[span['end']:]
        )
    return masked_text


# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================

def run_phase1_normalization(raw_message: str, room_id: str, msg_id: str) -> Dict[str, Any]:
    # 1. Canonicalization
    canon_text = sanitize_unicode(raw_message)
    canon_text = despace_isolated_characters(canon_text)
    canon_text = despace_path_separators(canon_text)
    canon_text = flatten_whitespace(canon_text)
    canon_text = convert_words_to_digits(canon_text)
    canon_text = substitute_verbal(canon_text)
    
    # 2. Parallel Structural Detection
    detected_spans = detect_patterns(canon_text)
    
    # 3. Mask Spans (if any exist)
    processed_text = mask_matched_spans(canon_text, detected_spans)
    
    # 4. Final Revert Node
    final_text = revert_verbal(processed_text)
    
    # 5. ALWAYS FORWARD TO PHASE 2
    return {
        "room_id": room_id,
        "msg_id": msg_id,
        "final_text": final_text,
        "spans_detected": detected_spans,
    }