"""sap3d-edit rejects inconsistent flags before loading any model or mesh."""

from sap3d.edit import cli

MISSING = "/nonexistent/sap3d"


def _exits(argv) -> bool:
    try:
        cli.main(argv + ["--out", MISSING + "/out"])
    except SystemExit as stop:
        return stop.code not in (0, None)
    return False


def test_benchmark_case_rejects_asset_flags():
    case = ["--bench-root", MISSING, "--case", "GSO/X/prompt_1", "--adapter", MISSING]
    assert _exits(case + ["--global"])
    assert _exits(case + ["--mask", "m.png"])
    assert _exits(["--case", "GSO/X/prompt_1", "--adapter", MISSING])


def test_asset_flag_combinations():
    asset = ["--source", MISSING + ".glb", "--edited-view", MISSING + ".png", "--adapter", MISSING]
    assert _exits(asset + ["--global", "--region", "r.yaml", "--mask", "m.png"])
    assert _exits(asset + ["--region", "r.yaml"])
    assert _exits(asset + ["--region", "r.yaml", "--mask", "m.png", "--orbit", "15", "-90"])
    assert _exits(asset + ["--region", "r.yaml", "--mask", "m.png", "--view", "3"])
    assert _exits(["--source", MISSING + ".glb", "--region", "r.yaml", "--mask", "m.png", "--adapter", MISSING])


def test_uniform_mode_rejects_region_inputs():
    asset = ["--source", MISSING + ".glb", "--edited-view", MISSING + ".png", "--adapter", MISSING,
             "--config", "configs/edit_global_uniform.yaml"]
    assert _exits(asset + ["--global"])
    assert _exits(asset + ["--mask", "m.png"])


def test_missing_adapter_and_output():
    assert _exits(["--bench-root", MISSING, "--case", "GSO/X/prompt_1"])
    assert _exits(["--source", MISSING + ".glb", "--edited-view", MISSING + ".png", "--global",
                   "--orbit", "15", "-90", "--adapters", MISSING])
    try:
        cli.main(["--source", "a.glb"])
    except SystemExit as stop:
        assert stop.code not in (0, None)
        return
    raise AssertionError("no --out was accepted")
