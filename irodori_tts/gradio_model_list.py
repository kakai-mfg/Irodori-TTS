"""Shared model choices for the editable Gradio checkpoint dropdowns."""

from pathlib import Path

import yaml

DEFAULT_MODEL_LIST_PATH = Path(__file__).resolve().parents[1] / "configs" / "gradio_models.yaml"


def load_model_choices(config_path: str | Path | None = None) -> list[tuple[str, str]]:
    source = DEFAULT_MODEL_LIST_PATH if config_path is None else Path(config_path).expanduser()
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if config_path is None:
            return []
        raise ValueError(f"Model list config not found: {source}") from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read model list config {source}: {exc}") from exc

    if config is None:
        return []
    if not isinstance(config, dict) or not isinstance(config.get("models"), list):
        raise ValueError(f"{source}: expected a 'models' list.")

    choices: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(config["models"], start=1):
        if isinstance(entry, str):
            checkpoint = name = entry
        elif isinstance(entry, dict):
            checkpoint = entry.get("checkpoint")
            name = entry.get("name", checkpoint)
        else:
            raise ValueError(f"{source}: model {index} must be a string or a mapping.")

        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"{source}: model {index} must have a non-empty checkpoint string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source}: model {index} must have a non-empty name string.")
        checkpoint, name = checkpoint.strip(), name.strip()
        if checkpoint not in seen:
            choices.append((name, checkpoint))
            seen.add(checkpoint)
    return choices
