import re
import numpy as np

# ============================================================
# SOURCE / PUBLISHER LEAK CLEANING
# ============================================================
SOURCE_LEAK_TERMS = [
    "the indian express",
    "indian express",
    "indian_express",
    "indianexpress",
    "reuters",
    "associated press",
    "ap news",
    "bbc",
    "cnn",
    "the guardian",
    "new york times",
]

SOURCE_LEAK_REGEX = re.compile(
    r"(?i)\b(?:"
    + "|".join(re.escape(term).replace(r"\ ", r"[\s_-]+") for term in SOURCE_LEAK_TERMS)
    + r")\b"
)

def remove_source_leaks(text):
    text = str(text)

    # remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove source names
    text = SOURCE_LEAK_REGEX.sub(" ", text)

    # Clean repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================================================
# PER-TEXT STRUCTURAL STATISTICS
# ============================================================
def compute_text_stats(texts):
    """Returns a dense float32 array of [word_count, avg_sent_len, lexical_diversity] per text."""
    stats = []
    for text in texts:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        word_count = len(words)
        sent_count = len(sentences)
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = (unique_words / word_count) * 100 if word_count > 0 else 0
        stats.append([word_count, avg_sent_len, lexical_diversity])
    return np.array(stats, dtype=np.float32)
