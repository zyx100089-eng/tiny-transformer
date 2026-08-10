import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer


def test_fit_builds_vocab():
    tok = CharTokenizer().fit("hello")
    assert tok.vocab_size == 4  # h, e, l, o
    assert set(tok.stoi.keys()) == {"h", "e", "l", "o"}


def test_encode_decode_roundtrip():
    tok = CharTokenizer().fit("hello world")
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_encode_produces_integers():
    tok = CharTokenizer().fit("abc")
    encoded = tok.encode("abc")
    assert all(isinstance(i, int) for i in encoded)
    assert len(encoded) == 3


def test_sorted_vocabulary():
    tok = CharTokenizer().fit("zab")
    chars = [tok.itos[i] for i in range(tok.vocab_size)]
    assert chars == ["a", "b", "z"]


def test_save_load(tmp_path):
    tok = CharTokenizer().fit("hello world")
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = CharTokenizer().load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.decode(tok2.encode("hello")) == "hello"


# --- BPE tests ---

def test_bpe_fit_reduces_token_count():
    text = "aaa bbb aaa bbb aaa bbb"
    tok = BPETokenizer().fit(text, num_merges=3)
    assert tok.vocab_size > 4
    encoded = tok.encode(text)
    assert len(encoded) < len(text)


def test_bpe_encode_decode_roundtrip():
    text = "hello world"
    tok = BPETokenizer().fit(text, num_merges=5)
    assert tok.decode(tok.encode(text)) == text


def test_bpe_vocab_contains_merges():
    text = "abababab"
    tok = BPETokenizer().fit(text, num_merges=1)
    assert "ab" in tok.vocab


def test_bpe_save_load(tmp_path):
    text = "hello world hello"
    tok = BPETokenizer().fit(text, num_merges=3)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    tok2 = BPETokenizer().load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.decode(tok2.encode(text)) == text


def test_bpe_unseen_character_maps_to_unk():
    """Unseen characters must map to <unk>, not silently substitute another char.

    Regression test: previously `encode('hz')` (z unseen) decoded to 'hh'
    because the fallback was `stoi.get(text[0], 0)` — the first character of
    the whole input was used for every unknown token.
    """
    tok = BPETokenizer().fit("hello world", num_merges=2)
    ids = tok.encode("hz")
    assert tok.decode(ids) == "h<unk>"


def test_bpe_unk_id_is_zero():
    """The <unk> token must occupy id 0 so it is reserved across save/load."""
    tok = BPETokenizer().fit("ab", num_merges=1)
    assert tok.unk_id == 0
    assert tok.stoi["<unk>"] == 0
    assert tok.vocab[0] == "<unk>"