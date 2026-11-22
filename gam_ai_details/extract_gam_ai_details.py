#!/usr/bin/env python3
"""Write the documentation record for every packaged pygamlab .gam_ai model.

Each file is loaded with pygamlab.ai_core.gam_ai.GAM_AI_MODEL. The saved
record contains the metadata printed by GAM_AI_MODEL.summary and the class
and parameters of the deserialized estimator.
"""

from __future__ import annotations

import io
import json
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pygamlab
from pygamlab.ai_core.gam_ai import GAM_AI_MODEL

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"


def jsonable(value):
    """Convert estimator parameters and metadata into JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return str(value)


def qualified_name(estimator) -> str:
    if estimator is None:
        return "None"
    return f"{type(estimator).__module__}.{type(estimator).__qualname__}"


def estimator_details(estimator) -> dict:
    """Describe the loaded object, unwrapping a saved search when present."""
    details = {"ml_model_class": qualified_name(estimator)}
    searched = estimator
    if hasattr(estimator, "best_estimator_"):
        details["search_best_score"] = jsonable(getattr(estimator, "best_score_", None))
        details["search_best_params"] = jsonable(getattr(estimator, "best_params_", {}))
        searched = estimator.best_estimator_
        details["best_estimator_class"] = qualified_name(searched)
    if hasattr(searched, "steps"):
        final_name, final_estimator = searched.steps[-1]
        details["pipeline_steps"] = [name for name, _step in searched.steps]
        details["final_step"] = final_name
        details["final_estimator_class"] = qualified_name(final_estimator)
        details["final_estimator_params"] = jsonable(final_estimator.get_params())
    elif hasattr(searched, "get_params"):
        details["estimator_params"] = jsonable(searched.get_params())
    return details


def load_model(path: Path) -> GAM_AI_MODEL:
    with redirect_stdout(io.StringIO()):
        return GAM_AI_MODEL(path.stem, base_dir=str(path.parent))


def model_record(path: Path) -> dict:
    loaded = load_model(path)
    metadata = {
        key: jsonable(value)
        for key, value in loaded.__dict__.items()
        if key not in {"ml_model", "file_path"}
    }
    record = {
        "file_name": path.name,
        **metadata,
        **estimator_details(loaded.ml_model),
    }
    return record


def is_blank(value) -> bool:
    if value is None or value == "" or value == {}:
        return True
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    return False


def accuracy_text(value) -> str:
    if isinstance(value, dict):
        rendered = ", ".join(f"{key} {item}" for key, item in value.items())
    elif isinstance(value, str) and value.startswith("(Pipeline"):
        rendered = value.rstrip(")").rsplit(",", 1)[-1].strip()
    elif value is None:
        rendered = ""
    else:
        rendered = str(value)
    return " ".join(rendered.replace("|", "/").split())


def write_markdown(records: list[dict], path: Path):
    lines = [
        "# pygamlab .gam_ai model details",
        "",
        f"pygamlab {pygamlab.__version__}. Each model was loaded with `GAM_AI_MODEL`.",
        "",
        "| Model | Type | Estimator | Recorded result |",
        "|---|---|---|---|",
    ]
    for record in records:
        estimator = record.get("final_estimator_class") or record["ml_model_class"]
        estimator = estimator.rsplit(".", 1)[-1]
        lines.append(
            "| {name} | {kind} | {estimator} | {result} |".format(
                name=record["model_name"],
                kind=record.get("model_type", ""),
                estimator=estimator,
                result=accuracy_text(record.get("best_accuracy")).replace("|", "/"),
            )
        )
    lines.extend(["", "## Model records", ""])
    fields = (
        "model_name",
        "model_type",
        "description",
        "author_name",
        "author_email",
        "trainer_name",
        "doi",
        "best_accuracy",
        "hyperparam_range",
        "best_params",
        "ml_model_class",
        "best_estimator_class",
        "pipeline_steps",
        "final_estimator_class",
        "final_estimator_params",
        "estimator_params",
        "search_best_score",
        "search_best_params",
    )
    for record in records:
        lines.append(f"### {record['model_name']}")
        lines.append("")
        for field in fields:
            if field not in record or is_blank(record[field]):
                continue
            value = record[field]
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, indent=2)
                lines.append(f"**{field}**")
                lines.append("")
                lines.append("```json")
                lines.append(rendered)
                lines.append("```")
            else:
                lines.append(f"**{field}:** {value}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(MODEL_DIR.glob("*.gam_ai"))
    records = [model_record(path) for path in paths]
    payload = {
        "pygamlab_version": pygamlab.__version__,
        "loader": "pygamlab.ai_core.gam_ai.GAM_AI_MODEL",
        "model_count": len(records),
        "models": records,
    }
    json_path = OUT_DIR / "model_details.json"
    markdown_path = OUT_DIR / "model_details.md"
    json_path.write_text(json.dumps(payload, indent=2))
    write_markdown(records, markdown_path)
    print(f"pygamlab {pygamlab.__version__}")
    print(f"{len(records)} models")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
