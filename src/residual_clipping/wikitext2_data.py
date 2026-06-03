"""WikiText-2 word-level language-modeling data helpers."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch


WIKITEXT2_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
WIKITEXT2_ARCHIVE = "wikitext-2-v1.zip"
WIKITEXT2_DIR = "wikitext-2"


class Dictionary:
    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


@dataclass(frozen=True)
class LanguageModelCorpus:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    dictionary: Dictionary

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)


def wikitext2_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / WIKITEXT2_DIR


def maybe_download_wikitext2(data_dir: str | Path, *, download: bool) -> Path:
    root = wikitext2_root(data_dir)
    train_file = root / "wiki.train.tokens"
    if train_file.exists():
        return root
    if not download:
        raise FileNotFoundError(
            f"WikiText-2 was not found at {root}. Re-run with --download or place "
            "wiki.train.tokens, wiki.valid.tokens, and wiki.test.tokens there."
        )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / WIKITEXT2_ARCHIVE
    urllib.request.urlretrieve(WIKITEXT2_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_path)
    if not train_file.exists():
        raise FileNotFoundError(f"Downloaded WikiText-2, but expected {train_file} to exist.")
    return root


def _read_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        words.extend(line.strip().split())
        words.append("<eos>")
    return words


def _tokenize(path: Path, dictionary: Dictionary, *, add_new_words: bool) -> torch.Tensor:
    ids: list[int] = []
    for word in _read_words(path):
        if add_new_words:
            ids.append(dictionary.add_word(word))
        else:
            ids.append(dictionary.word2idx.get(word, dictionary.word2idx["<unk>"]))
    return torch.tensor(ids, dtype=torch.long)


def _make_fake_corpus(fake_token_count: int, fake_vocab_size: int) -> LanguageModelCorpus:
    dictionary = Dictionary()
    for index in range(fake_vocab_size):
        dictionary.add_word(f"token_{index}")
    dictionary.add_word("<eos>")
    dictionary.add_word("<unk>")

    def make_split(offset: int, length: int) -> torch.Tensor:
        values = [(offset + index) % fake_vocab_size for index in range(length)]
        return torch.tensor(values, dtype=torch.long)

    train_size = max(fake_token_count, 80)
    valid_size = max(fake_token_count // 4, 40)
    test_size = max(fake_token_count // 4, 40)
    return LanguageModelCorpus(
        train=make_split(0, train_size),
        valid=make_split(3, valid_size),
        test=make_split(7, test_size),
        dictionary=dictionary,
    )


def load_corpus(
    dataset_name: str,
    data_dir: str | Path,
    *,
    download: bool,
    fake_token_count: int,
    fake_vocab_size: int,
) -> LanguageModelCorpus:
    if dataset_name == "fake_wikitext2":
        return _make_fake_corpus(fake_token_count, fake_vocab_size)
    if dataset_name != "wikitext2":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    root = maybe_download_wikitext2(data_dir, download=download)
    dictionary = Dictionary()
    dictionary.add_word("<unk>")
    train = _tokenize(root / "wiki.train.tokens", dictionary, add_new_words=True)
    valid = _tokenize(root / "wiki.valid.tokens", dictionary, add_new_words=False)
    test = _tokenize(root / "wiki.test.tokens", dictionary, add_new_words=False)
    return LanguageModelCorpus(train=train, valid=valid, test=test, dictionary=dictionary)


def batchify(data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    num_batches = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source: torch.Tensor, index: int, bptt: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, len(source) - 1 - index)
    data = source[index : index + seq_len]
    target = source[index + 1 : index + 1 + seq_len].reshape(-1)
    return data, target
