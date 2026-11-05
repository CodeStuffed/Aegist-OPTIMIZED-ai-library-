"""The .nx brain file: round trips, paged loading straight from the file, and tamper detection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402

from aegist.model import AegistConfig, AegistLM  # noqa: E402
from aegist.nx import MAGIC, NxFile, load_model, save_model  # noqa: E402


def tiny_tokenizer_json() -> str:
    tok = Tokenizer(models.WordLevel({"hi": 0, "there": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    return tok.to_str()


def tiny(quant: bool):
    torch.manual_seed(0)
    cfg = AegistConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8, ffn_hidden=64,
                       max_seq_len=256, core_grid=(4, 4), core_rank=4, cores_active=2)
    m = AegistLM(cfg)
    with torch.no_grad():
        m.blocks[0].cores.down.normal_(std=0.3)  # layer 0 trained, layer 1 still silent
    if quant:
        m.quantize_base_int8()
    return m.eval()


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("train", [False, True])
def test_round_trip_gives_identical_outputs(tmp_path, quant, train):
    m = tiny(quant)
    path = str(tmp_path / "t.nx")
    hdr = save_model(path, m, tiny_tokenizer_json(), notice="test notice", license_text="LICENSE TEXT",
                     lineage=[{"event": "born"}])
    assert hdr["cores_silent"] == [False, True]
    x = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m(x)["logits"]
    m2, tok, nx = load_model(path, device="cpu", slots=3, train=train)
    assert all(b.cores.paged != train for b in m2.blocks)
    if train:
        assert all(p.requires_grad for p in m2.core_parameters())
        assert not any(p.requires_grad for p in m2.base_parameters())
    m2.eval()
    assert [b.cores.silent for b in m2.blocks] == [False, True]
    with torch.no_grad():
        got = m2(x)["logits"]
    assert torch.allclose(got, ref, atol=1e-5)  # paged cores may take the dispatch path
    assert nx.blob("LICENSE") == b"LICENSE TEXT"
    assert nx.header["notice"] == "test notice" and nx.header["lineage"] == [{"event": "born"}]
    assert tok.im_start is not None
    assert all(nx.verify().values())
    nx.close()


def test_resaving_a_paged_model_keeps_every_core(tmp_path):
    m = tiny(quant=True)
    a, b = str(tmp_path / "a.nx"), str(tmp_path / "b.nx")
    save_model(a, m, tiny_tokenizer_json())
    m2, _, nx = load_model(a, device="cpu", slots=2)
    save_model(b, m2, tiny_tokenizer_json())
    nx.close()
    with NxFile(a) as fa, NxFile(b) as fb:
        assert sorted(fa.names()) == sorted(fb.names())
        for n in fa.names():
            assert torch.equal(fa.tensor(n), fb.tensor(n)), n
        assert fa.header["sha256"] == fb.header["sha256"]


def test_tensors_are_aligned_and_single_cores_readable(tmp_path):
    m = tiny(quant=False)
    path = str(tmp_path / "t.nx")
    save_model(path, m, tiny_tokenizer_json())
    with NxFile(path) as nx:
        assert all(meta["offset"] % 64 == 0 for meta in nx.header["tensors"].values())
        assert nx.data_start % 64 == 0
        core = nx.core(0, 5)
        assert torch.equal(core["down"], m.blocks[0].cores.down[5].detach())
        assert set(nx.names("cores")) == {f"blocks.{i}.cores.{k}" for i in range(2) for k in ("gate", "up", "down")}
        assert set(nx.names("engine")) == {f"blocks.{i}.engine.key{j}.weight" for i in range(2) for j in (1, 2)}


def test_corruption_is_detected(tmp_path):
    m = tiny(quant=False)
    path = str(tmp_path / "t.nx")
    save_model(path, m, tiny_tokenizer_json())
    with NxFile(path) as nx:
        meta = nx.header["tensors"]["blocks.0.cores.down"]
        pos = nx.data_start + meta["offset"] + 10
    with open(path, "r+b") as f:
        f.seek(pos)
        byte = f.read(1)
        f.seek(pos)
        f.write(bytes([byte[0] ^ 0xFF]))
    with NxFile(path) as nx:
        assert nx.verify() == {"base": True, "engine": True, "cores": False}


def test_rejects_other_files(tmp_path):
    p = tmp_path / "not.nx"
    p.write_bytes(b"PK\x03\x04" + b"\0" * 64)
    with pytest.raises(ValueError):
        NxFile(str(p))
    assert MAGIC == b"AEGISTNX"
