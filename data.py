"""
Dataset pour des paires (audio, texte).

Supporte tokenisation par mots, caractères, ou les deux (lexique multi-niveau).
"""

import glob
import io
import json
import os
from collections import Counter
from typing import Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from align import AlignmentStore, utterance_key

WORD_PREFIX = "w:"
CHAR_PREFIX = "c:"


def _resample(waveform: torch.Tensor, sr: int, sample_rate: int) -> torch.Tensor:
    if sr == sample_rate:
        return waveform
    # Interpolation linéaire (évite d'importer torchaudio → segfault CUDA sur achille)
    n = int(round(waveform.numel() * sample_rate / sr))
    if n <= 0:
        return waveform
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1).float(), size=n, mode="linear", align_corners=False
    ).view(-1)


def load_wav_file(path: str, sample_rate: int = 16_000,
                   max_audio_sec: float = 30.0) -> torch.Tensor:
    """Charge un fichier audio (FLAC/WAV) via soundfile — évite segfault CUDA torchaudio."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(audio)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)
    waveform = _resample(waveform, int(sr), sample_rate)
    max_samples = int(max_audio_sec * sample_rate)
    return waveform[:max_samples].contiguous()


def _word_key(word: str) -> str:
    return f"{WORD_PREFIX}{word}"


def _char_key(char: str) -> str:
    return f"{CHAR_PREFIX}{char}"


def unit_type(key: str) -> str:
    if key.startswith(WORD_PREFIX):
        return "word"
    if key.startswith(CHAR_PREFIX):
        return "char"
    return "other"


def unit_display(key: str) -> str:
    if key.startswith(WORD_PREFIX):
        return key[len(WORD_PREFIX):]
    if key.startswith(CHAR_PREFIX):
        return key[len(CHAR_PREFIX):]
    return key


def build_vocab_from_parquet(parquet_dir: str, split: str = "train",
                              max_files: int = None,
                              include_words: bool = True,
                              include_chars: bool = True,
                              min_word_freq: int = 2,
                              max_words: int = 10000) -> dict:
    """Vocabulaire multi-niveau : w:mot (fréquents) + c:lettre."""
    files = _resolve_parquet_files(parquet_dir, split, max_files)
    word_counts = Counter()
    chars = set()
    for fp in files:
        table = pq.read_table(fp, columns=["transcription"])
        for text in table["transcription"].to_pylist():
            if include_words:
                word_counts.update(text.split())
            if include_chars:
                chars.update(text)

    symbol_to_id = {}
    idx = 1
    if include_chars:
        for c in sorted(chars):
            symbol_to_id[_char_key(c)] = idx
            idx += 1
    if include_words:
        frequent = [w for w, c in word_counts.most_common()
                    if c >= min_word_freq][:max_words]
        for w in sorted(frequent):
            key = _word_key(w)
            if key not in symbol_to_id:
                symbol_to_id[key] = idx
                idx += 1
    return symbol_to_id


def save_vocab_meta(symbol_to_id: dict, path: str):
    meta = {
        "symbol_to_id": symbol_to_id,
        "unit_types": {k: unit_type(k) for k in symbol_to_id},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "symbol_to_id" in data:
        symbol_to_id = data["symbol_to_id"]
        unit_types = data.get("unit_types", {k: unit_type(k) for k in symbol_to_id})
    else:
        symbol_to_id = data
        unit_types = {k: unit_type(k) for k in symbol_to_id}
    assert 0 not in symbol_to_id.values(), "L'id 0 est réservé au padding"
    return symbol_to_id, unit_types


def tokenize_text(text: str, symbol_to_id: dict, mode: str = "words",
                  max_len: int = 511) -> list:
    """
    mode='words'  -> séquence de mots  (entraînement principal)
    mode='chars'  -> séquence de caractères
    mode='both'   -> mots puis caractères de la phrase (lexique riche)
    """
    ids = []
    if mode in ("words", "both"):
        for w in text.split():
            key = _word_key(w)
            if key in symbol_to_id:
                ids.append(symbol_to_id[key])
    if mode in ("chars", "both"):
        for c in text:
            key = _char_key(c)
            if key in symbol_to_id:
                ids.append(symbol_to_id[key])
    return ids[:max_len]


def _resolve_parquet_files(parquet_dir: str, split: str, max_files: int = None) -> list:
    patterns = {
        "train": "ewe-train-*.parquet",
        "test": "ewe_test_clean.parquet",
        "val": "ewe_validation_clean.parquet",
        "validation": "ewe_validation_clean.parquet",
    }
    pattern = patterns.get(split, split)
    path = os.path.join(parquet_dir, pattern)
    files = sorted(glob.glob(path))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Aucun fichier parquet trouvé : {path}")
    return files


def load_audio_from_bytes(audio_bytes: bytes, sample_rate: int = 16_000,
                          max_audio_sec: float = 30.0) -> Optional[torch.Tensor]:
    """Charge audio depuis bytes parquet (wav/flac/mp3) ; None si illisible."""
    max_samples = int(max_audio_sec * sample_rate)

    # 1) soundfile (wav/flac/ogg)
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        waveform = torch.from_numpy(audio)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        waveform = _resample(waveform, int(sr), sample_rate)
        return waveform[:max_samples].contiguous()
    except Exception:
        pass

    # 2) librosa (mp3 via audioread — corpus éwé)
    try:
        import librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
        waveform = torch.from_numpy(audio.astype("float32"))
        return waveform[:max_samples].contiguous()
    except Exception:
        pass

    # 3) fichier temporaire + librosa (certains backends refusent BytesIO)
    try:
        import tempfile
        import librosa
        suffix = ".mp3" if audio_bytes[:3] == b"ID3" else ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio, sr = librosa.load(tmp.name, sr=sample_rate, mono=True)
        waveform = torch.from_numpy(audio.astype("float32"))
        return waveform[:max_samples].contiguous()
    except Exception:
        pass

    return None


def _load_audio_from_bytes(audio_bytes: bytes, sample_rate: int,
                            max_audio_sec: float = 30.0) -> torch.Tensor:
    waveform = load_audio_from_bytes(audio_bytes, sample_rate, max_audio_sec)
    if waveform is None:
        raise ValueError("Format audio non reconnu")
    return waveform


class AudioTextDataset(Dataset):
    def __init__(self, manifest_path: str, symbol_to_id: dict, sample_rate: int = 16_000,
                 token_mode: str = "words"):
        self.examples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.symbol_to_id = symbol_to_id
        self.sample_rate = sample_rate
        self.token_mode = token_mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        waveform = load_wav_file(ex["audio_path"], self.sample_rate, max_audio_sec=30.0)
        token_ids = tokenize_text(ex["text"], self.symbol_to_id, self.token_mode)
        return {
            "waveform": waveform,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": ex["text"],
        }


def _resolve_librispeech_split(root: str, split: str) -> str:
    """Retourne le chemin du split LibriSpeech (ex. train-clean-100)."""
    path = os.path.join(root, split)
    if os.path.isdir(path):
        return path
    alt = os.path.join(root, "LibriSpeech", split)
    if os.path.isdir(alt):
        return alt
    raise FileNotFoundError(f"Split LibriSpeech introuvable : {split} sous {root}")


def iter_librispeech_utterances(librispeech_root: str, split: str):
    """Parcourt (audio_path, utterance_id, text) pour un split LibriSpeech."""
    split_dir = _resolve_librispeech_split(librispeech_root, split)
    for trans_path in sorted(glob.glob(os.path.join(split_dir, "*", "*", "*.trans.txt"))):
        chapter_dir = os.path.dirname(trans_path)
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uid, text = line.split(" ", 1)
                audio_path = os.path.join(chapter_dir, f"{uid}.flac")
                if os.path.isfile(audio_path):
                    yield audio_path, uid, text


def build_vocab_from_librispeech(librispeech_root: str, split: str = "train-clean-100",
                                  include_words: bool = True, include_chars: bool = True,
                                  min_word_freq: int = 5, max_words: int = 20000) -> dict:
    """Vocabulaire multi-niveau depuis LibriSpeech ([SLR12](https://www.openslr.org/12))."""
    word_counts = Counter()
    chars = set()
    for _, _, text in iter_librispeech_utterances(librispeech_root, split):
        norm = text.lower()
        if include_words:
            word_counts.update(norm.split())
        if include_chars:
            chars.update(norm)
    symbol_to_id = {}
    idx = 1
    if include_chars:
        for c in sorted(chars):
            symbol_to_id[_char_key(c)] = idx
            idx += 1
    if include_words:
        frequent = [w for w, c in word_counts.most_common()
                    if c >= min_word_freq][:max_words]
        for w in sorted(frequent):
            key = _word_key(w)
            if key not in symbol_to_id:
                symbol_to_id[key] = idx
                idx += 1
    return symbol_to_id


class LibriSpeechDataset(Dataset):
    """Corpus LibriSpeech : fichiers .flac + .trans.txt par chapitre."""

    def __init__(self, librispeech_root: str, symbol_to_id: dict, split: str = "train-clean-100",
                 sample_rate: int = 16_000, token_mode: str = "words",
                 max_audio_sec: float = 30.0, alignment_store: Optional[AlignmentStore] = None):
        self.symbol_to_id = symbol_to_id
        self.sample_rate = sample_rate
        self.token_mode = token_mode
        self.max_audio_sec = max_audio_sec
        self.alignment_store = alignment_store
        self.examples = list(iter_librispeech_utterances(librispeech_root, split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        audio_path, utterance_id, text = self.examples[idx]
        text = text.lower()
        waveform = load_wav_file(audio_path, self.sample_rate, self.max_audio_sec)
        token_ids = tokenize_text(text, self.symbol_to_id, self.token_mode)
        out = {
            "waveform": waveform,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "utterance_id": utterance_id,
        }
        if self.alignment_store is not None:
            targets = self.alignment_store.get(utterance_id, 0)
            if targets is not None:
                out["seg_targets"] = torch.tensor(targets, dtype=torch.long)
        return out


class ParquetAudioTextDataset(Dataset):
    def __init__(self, parquet_dir: str, symbol_to_id: dict, sample_rate: int = 16_000,
                 split: str = "train", max_files: int = None, token_mode: str = "words",
                 max_audio_sec: float = 30.0, alignment_store: Optional[AlignmentStore] = None):
        self.symbol_to_id = symbol_to_id
        self.sample_rate = sample_rate
        self.token_mode = token_mode
        self.max_audio_sec = max_audio_sec
        self.alignment_store = alignment_store
        self.examples = []
        self._tables = {}

        for fp in _resolve_parquet_files(parquet_dir, split, max_files):
            table = pq.read_table(fp, columns=["transcription"])
            for i, text in enumerate(table["transcription"].to_pylist()):
                self.examples.append((fp, i, text))

    def __len__(self):
        return len(self.examples)

    def _get_audio_bytes(self, parquet_path: str, row_idx: int) -> bytes:
        if parquet_path not in self._tables:
            self._tables[parquet_path] = pq.read_table(parquet_path, columns=["audio"])
        audio_col = self._tables[parquet_path]["audio"]
        return audio_col[row_idx]["bytes"].as_py()

    def __getitem__(self, idx):
        # Skip les audios illisibles (mp3 corrompus, etc.)
        n = len(self.examples)
        for offset in range(n):
            j = (idx + offset) % n
            parquet_path, row_idx, text = self.examples[j]
            try:
                audio_bytes = self._get_audio_bytes(parquet_path, row_idx)
                waveform = _load_audio_from_bytes(
                    audio_bytes, self.sample_rate, self.max_audio_sec,
                )
            except Exception:
                continue
            token_ids = tokenize_text(text, self.symbol_to_id, self.token_mode)
            out = {
                "waveform": waveform,
                "token_ids": torch.tensor(token_ids, dtype=torch.long),
                "text": text,
                "parquet_path": parquet_path,
                "row_idx": row_idx,
            }
            if self.alignment_store is not None:
                targets = self.alignment_store.get(parquet_path, row_idx)
                if targets is not None:
                    out["seg_targets"] = torch.tensor(targets, dtype=torch.long)
            return out
        raise RuntimeError("Aucun audio lisible dans le dataset")


def collate_fn(batch, symbol_to_id: dict = None, with_chars: bool = False):
    wave_lens = [b["waveform"].size(0) for b in batch]
    tok_lens = [b["token_ids"].size(0) for b in batch]
    max_wave = max(wave_lens)
    max_tok = max(1, max(tok_lens))

    waveforms = torch.zeros(len(batch), max_wave)
    audio_mask = torch.zeros(len(batch), max_wave, dtype=torch.long)
    token_ids = torch.zeros(len(batch), max_tok, dtype=torch.long)
    text_mask = torch.zeros(len(batch), max_tok, dtype=torch.long)

    for i, b in enumerate(batch):
        wl, tl = wave_lens[i], tok_lens[i]
        waveforms[i, :wl] = b["waveform"]
        audio_mask[i, :wl] = 1
        if tl > 0:
            token_ids[i, :tl] = b["token_ids"]
            text_mask[i, :tl] = 1

    out = {
        "waveforms": waveforms,
        "audio_attention_mask": audio_mask,
        "token_ids": token_ids,
        "text_attention_mask": text_mask,
    }

    if with_chars and symbol_to_id is not None and "text" in batch[0]:
        char_lens = []
        char_seqs = []
        for b in batch:
            cids = tokenize_text(b["text"], symbol_to_id, mode="chars")
            char_seqs.append(cids)
            char_lens.append(len(cids))
        max_c = max(1, max(char_lens))
        char_ids = torch.zeros(len(batch), max_c, dtype=torch.long)
        char_mask = torch.zeros(len(batch), max_c, dtype=torch.long)
        for i, cids in enumerate(char_seqs):
            if cids:
                char_ids[i, :len(cids)] = torch.tensor(cids, dtype=torch.long)
                char_mask[i, :len(cids)] = 1
        out["char_token_ids"] = char_ids
        out["char_attention_mask"] = char_mask

    if any("seg_targets" in b for b in batch):
        seg_lens = [b["seg_targets"].size(0) for b in batch if "seg_targets" in b]
        max_seg = max(seg_lens) if seg_lens else 1
        seg_targets = torch.full((len(batch), max_seg), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            if "seg_targets" in b:
                sl = b["seg_targets"].size(0)
                seg_targets[i, :sl] = b["seg_targets"]
        out["seg_targets"] = seg_targets

    return out
