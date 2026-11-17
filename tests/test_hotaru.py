from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model import build_model
from model.hotaru.hotaru import EPSILON, MAX_LENGTH, Hotaru, make_score_mask
from model.yokoyama.yokoyama import Yokoyama, make_score_mask as yokoyama_mask
from utils.exceptions import Error
from utils.validator.load import load_model


def test_hotaru_matches_yokoyama() -> None:
    original = Yokoyama().eval()
    copied = Hotaru().eval()
    copied.load_state_dict(original.state_dict(), strict=True)
    ids = torch.randint(1, 4096, (2, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    with torch.no_grad():
        torch.testing.assert_close(copied(ids, mask), original(ids, mask), rtol=0, atol=0)
    assert torch.equal(make_score_mask(mask), yokoyama_mask(mask))


@pytest.mark.parametrize("length", [8, MAX_LENGTH])
def test_hotaru_configuration_and_backward(length: int) -> None:
    model = build_model("hotaru", None)
    assert isinstance(model, Hotaru)
    assert model.embedder.position.num_embeddings == MAX_LENGTH
    assert all(norm.eps == EPSILON for norm in model.norms)
    ids = torch.randint(1, 4096, (1, length))
    logits = model(ids, torch.ones_like(ids))
    assert logits.shape == (1, length, 4096)
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(Error):
        model(torch.ones((1, MAX_LENGTH + 1), dtype=torch.long), torch.ones((1, MAX_LENGTH + 1), dtype=torch.long))


def test_hotaru_validator_and_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "best.pth"
    model = build_model("hotaru", None)
    torch.save(model.state_dict(), weights)
    config = SimpleNamespace(
        run=SimpleNamespace(seed=0, device="cpu"),
        data=SimpleNamespace(max_length=12, tokenizer_dir="model/weight/hotaru"),
        model=SimpleNamespace(name="hotaru", checkpoint=str(weights)),
    )
    loaded, _, _ = load_model(config)
    assert isinstance(loaded, Hotaru)
    assert loaded.embedder.position.num_embeddings == MAX_LENGTH
    assert all(norm.eps == EPSILON for norm in loaded.norms)
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())


def test_hotaru_causality() -> None:
    model = Hotaru().eval()
    ids = torch.randint(1, 4096, (1, 12))
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 4096
    with torch.no_grad():
        torch.testing.assert_close(model(ids, torch.ones_like(ids))[:, :6], model(changed, torch.ones_like(ids))[:, :6], rtol=0, atol=0)
