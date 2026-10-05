"""The knee rule, against recorded token sets for fixed synthetic score matrices."""

import numpy as np

from sap3d.edit.knee import bend_indices, knee_region

WIDTH, SENSITIVITY = 51, 1.0


def knee_rows() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = 240
    a = np.exp(-np.arange(n) / 25.0)[rng.permutation(n)] + rng.uniform(0, 0.02, n)
    b = np.where(np.arange(n) < 30, 1.0, 0.05)[rng.permutation(n)]
    c = np.full(n, 0.3)
    d = (1.0 / (1.0 + np.arange(n)))[rng.permutation(n)]
    return np.stack([a, b, c, d])


EXPECTED_0_8344 = [
    0, 1, 3, 4, 7, 8, 9, 10, 12, 13, 15, 17, 21, 22, 25, 27, 29, 33, 35, 36, 37, 38, 40, 41, 42, 43,
    45, 46, 50, 51, 54, 57, 58, 60, 61, 63, 64, 65, 66, 70, 72, 73, 77, 78, 83, 85, 86, 87, 91, 94,
    96, 97, 98, 99, 100, 104, 105, 107, 108, 111, 113, 117, 118, 119, 120, 125, 126, 127, 131, 132,
    133, 136, 137, 139, 140, 141, 142, 143, 145, 147, 151, 152, 154, 155, 156, 157, 158, 159, 161,
    163, 165, 166, 167, 169, 170, 171, 172, 173, 175, 178, 182, 184, 186, 187, 189, 191, 192, 194,
    195, 197, 198, 199, 200, 202, 204, 206, 207, 209, 211, 213, 214, 216, 220, 222, 223, 224, 226,
    229, 231, 233, 234, 235, 238,
]
EXPECTED_2_5394 = [
    0, 7, 13, 17, 21, 25, 29, 33, 35, 36, 37, 38, 40, 41, 42, 45, 51, 54, 57, 58, 61, 63, 65, 66,
    72, 73, 77, 78, 85, 91, 94, 96, 97, 99, 105, 107, 108, 117, 118, 119, 120, 127, 132, 137, 139,
    140, 141, 142, 145, 147, 152, 155, 156, 157, 161, 163, 165, 166, 167, 169, 170, 171, 172, 173,
    175, 178, 184, 186, 187, 189, 191, 194, 195, 197, 198, 199, 200, 202, 206, 211, 213, 216, 222,
    224, 226, 229, 234, 235, 238,
]
EXPECTED_1_0 = [
    0, 3, 4, 7, 8, 9, 10, 12, 13, 15, 17, 21, 22, 25, 27, 29, 33, 35, 36, 37, 38, 40, 41, 42, 43,
    45, 46, 50, 51, 54, 57, 58, 60, 61, 63, 65, 66, 70, 72, 73, 77, 78, 83, 85, 86, 91, 94, 96, 97,
    98, 99, 100, 104, 105, 107, 108, 111, 117, 118, 119, 120, 125, 126, 127, 131, 132, 133, 137,
    139, 140, 141, 142, 143, 145, 147, 152, 154, 155, 156, 157, 159, 161, 163, 165, 166, 167, 169,
    170, 171, 172, 173, 175, 178, 184, 186, 187, 189, 191, 192, 194, 195, 197, 198, 199, 200, 202,
    204, 206, 209, 211, 213, 214, 216, 222, 223, 224, 226, 229, 231, 233, 234, 235, 238,
]
EXPECTED_0_8344_FP16 = [
    0, 1, 3, 4, 7, 8, 9, 10, 12, 13, 15, 17, 21, 22, 25, 27, 29, 33, 35, 36, 37, 38, 40, 41, 42, 43,
    45, 46, 50, 51, 54, 57, 58, 60, 61, 63, 64, 65, 66, 70, 72, 73, 77, 78, 83, 85, 86, 87, 91, 94,
    96, 97, 98, 99, 100, 104, 105, 107, 108, 111, 113, 117, 118, 119, 120, 125, 126, 127, 131, 132,
    133, 136, 137, 139, 140, 141, 142, 143, 145, 147, 151, 152, 154, 155, 156, 157, 158, 159, 161,
    163, 165, 166, 167, 169, 170, 171, 172, 173, 175, 178, 182, 184, 186, 187, 189, 191, 192, 194,
    195, 197, 198, 199, 200, 202, 204, 206, 207, 209, 211, 213, 214, 216, 220, 222, 223, 224, 226,
    229, 231, 233, 234, 235, 238,
]


def test_union_over_rows_matches_recorded_values():
    rows = knee_rows()
    assert sorted(knee_region(rows, 0.8344, WIDTH, SENSITIVITY)) == EXPECTED_0_8344
    assert sorted(knee_region(rows, 2.5394, WIDTH, SENSITIVITY)) == EXPECTED_2_5394
    assert sorted(knee_region(rows, 1.0, WIDTH, SENSITIVITY)) == EXPECTED_1_0


def test_float16_scores_match_recorded_values():
    assert sorted(knee_region(knee_rows().astype(np.float16), 0.8344, WIDTH, SENSITIVITY)) == EXPECTED_0_8344_FP16


def test_per_row_sizes():
    rows = knee_rows()
    sizes = [len(knee_region(rows[i:i + 1], 0.8344, WIDTH, SENSITIVITY)) for i in range(4)]
    assert sizes == [80, 30, 0, 55], sizes


def test_flat_row_admits_nothing():
    assert knee_region(np.full((3, 200), 0.25), 0.8344, WIDTH, SENSITIVITY) == set()


def test_sharp_step_bend_is_found_after_the_step():
    curve = np.concatenate([np.ones(100), np.zeros(200)])[None, :]
    index, flat = bend_indices(curve, WIDTH, SENSITIVITY)
    assert not flat[0] and 100 <= index[0] <= 100 + WIDTH // 2 + 1, index


def test_rejects_bad_inputs():
    rows = knee_rows()
    for bad in (-rows, np.where(rows > 0.5, np.nan, rows)):
        try:
            knee_region(bad, 0.8344, WIDTH, SENSITIVITY)
        except ValueError:
            continue
        raise AssertionError("invalid scores were accepted")
    try:
        knee_region(rows, 0.0, WIDTH, SENSITIVITY)
    except ValueError:
        return
    raise AssertionError("a zero knee factor was accepted")
