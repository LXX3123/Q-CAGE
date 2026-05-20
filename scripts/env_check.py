from __future__ import annotations

import importlib
import json
import platform


PACKAGES = [
    "torch",
    "transformers",
    "diffusers",
    "accelerate",
    "datasets",
    "PIL",
    "yaml",
]


def main() -> None:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "cuda": {},
    }
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            result["packages"][package] = version
        except Exception as exc:
            result["packages"][package] = f"missing: {exc}"

    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        result["cuda"] = {"error": str(exc)}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

