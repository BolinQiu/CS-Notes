import pathlib
import torch
from pathlib import Path



def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
) -> None:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
) -> torch.nn.Module:
    target_dir_path = Path(target_dir)
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Loading model from: {model_save_path}")
    model.load_state_dict(torch.load(f=model_save_path))

    return model
