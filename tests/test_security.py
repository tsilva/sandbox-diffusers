from __future__ import annotations

from importlib.metadata import version

import pytest
from diffusers.utils import dynamic_modules_utils
from packaging.version import Version


def test_diffusers_contains_remote_code_trust_fixes() -> None:
    assert Version(version("diffusers")) >= Version("0.38.0")


def test_untrusted_local_custom_code_is_rejected_before_execution(tmp_path) -> None:
    sentinel = tmp_path / "custom-code-executed"
    module_path = tmp_path / "malicious_pipeline.py"
    module_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed')\n"
        "class MaliciousPipeline:\n"
        "    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        dynamic_modules_utils.get_class_from_dynamic_module(
            tmp_path,
            module_path.name,
            class_name="MaliciousPipeline",
            local_files_only=True,
            trust_remote_code=False,
        )

    assert not sentinel.exists()


def test_untrusted_remote_custom_code_is_rejected_before_download(monkeypatch) -> None:
    download_attempted = False

    def fail_if_called(*args, **kwargs):
        nonlocal download_attempted
        download_attempted = True
        raise AssertionError("untrusted code must be rejected before a Hub download")

    monkeypatch.setattr(dynamic_modules_utils, "hf_hub_download", fail_if_called)

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        dynamic_modules_utils.get_cached_module_file(
            "attacker/cross-repository-pipeline",
            "pipeline.py",
            trust_remote_code=False,
        )

    assert download_attempted is False
