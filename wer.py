"""WER / CER pour évaluation ASR."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _edit_distance(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).replace(" ", "")
    h = normalize_text(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(list(r), list(h)) / len(r)


def aggregate_wer(pairs: list) -> dict:
    """pairs: [(ref, hyp), ...]"""
    if not pairs:
        return {"wer": 0.0, "cer": 0.0, "n": 0}
    w = sum(wer(r, h) for r, h in pairs) / len(pairs)
    c = sum(cer(r, h) for r, h in pairs) / len(pairs)
    return {"wer": w, "cer": c, "n": len(pairs)}
