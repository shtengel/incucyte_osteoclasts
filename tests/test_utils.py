import numpy as np
import pytest

from utils.utils import extract_incucyte_info, combine_image_statistics


def _make_stats(name, coverage_ratio=0.5):
    """Build a minimal image stats dict with a synthetic filtered segmentation."""
    mask = np.zeros((10, 10), dtype=np.int32)
    rows = int(10 * coverage_ratio)
    if rows > 0:
        mask[:rows, :] = 1
    return {
        "image_name": name,
        "filtered_segmentation": mask,
        "num_cells": 10,
        "cells_touching": 2,
        "mean_area": 100.0,
        "mean_perimeter": 40.0,
        "plate_coverage_percent": coverage_ratio,
    }


def test_extract_incucyte_info_valid():
    info = extract_incucyte_info("VID167_E7_3_02d18h00m")
    assert info["key"] == "VID167_E7_02d18h00m"
    assert info["plateNumber"] == "E7"
    assert info["position"] == "center"
    assert info["time"] == "02d18h00m"


def test_extract_incucyte_info_non_incucyte():
    info = extract_incucyte_info("some_random_image.png")
    assert info["key"] is None
    assert info["position"] is None
    assert info["plateNumber"] is None
    assert info["time"] is None


def test_extract_incucyte_info_wrong_underscore_count():
    info = extract_incucyte_info("VID167_E7_02d18h00m")
    assert info["key"] is None
    assert info["position"] is None


def test_combine_image_statistics_full_group():
    stats = [_make_stats(f"VID1_E7_{i}_t", coverage_ratio=0.5) for i in range(1, 6)]
    results = combine_image_statistics(stats)

    assert len(results) == 1
    row = results[0]
    assert row["image_name"] == "VID1_E7_t"
    assert row["num_cells"] == 50
    assert row["cells_touching"] == 10
    assert row["mean_area"] == 100.0
    assert row["mean_perimeter"] == 40.0
    assert abs(row["plate_coverage_percent"] - 0.5) < 1e-6


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_combine_image_statistics_partial_group(count):
    stats = [_make_stats(f"VID1_E7_{i}_t", coverage_ratio=0.5) for i in range(1, count + 1)]
    results = combine_image_statistics(stats)

    assert len(results) == 1
    row = results[0]
    assert row["image_name"] == "VID1_E7_t"
    assert row["num_cells"] == 10 * count
    assert row["cells_touching"] == 2 * count
    assert abs(row["plate_coverage_percent"] - 0.5) < 1e-6


def test_combine_image_statistics_zero_present_tiles():
    stats = [
        {
            "image_name": "VID1_E7_1_t",
            "filtered_segmentation": None,
            "num_cells": 0,
            "cells_touching": 0,
            "mean_area": 0.0,
            "mean_perimeter": 0.0,
            "plate_coverage_percent": 0.0,
        },
        {
            "image_name": "VID1_E7_3_t",
            "filtered_segmentation": None,
            "num_cells": 0,
            "cells_touching": 0,
            "mean_area": 0.0,
            "mean_perimeter": 0.0,
            "plate_coverage_percent": 0.0,
        },
    ]
    results = combine_image_statistics(stats)

    assert len(results) == 1
    assert results[0]["image_name"] == "VID1_E7_t"
    assert results[0]["plate_coverage_percent"] == 0.0


def test_combine_image_statistics_non_incucyte():
    stats = [
        _make_stats("random_image_1.png", coverage_ratio=0.3),
        _make_stats("random_image_2.png", coverage_ratio=0.7),
    ]
    results = combine_image_statistics(stats)

    assert len(results) == 2
    names = {row["image_name"] for row in results}
    assert names == {"random_image_1.png", "random_image_2.png"}


def test_combine_image_statistics_mixed_group():
    stats = [
        _make_stats("VID1_E7_1_t", coverage_ratio=0.5),
        _make_stats("VID1_E7_3_t", coverage_ratio=0.5),
        _make_stats("random_image.png", coverage_ratio=0.4),
    ]
    results = combine_image_statistics(stats)

    assert len(results) == 2
    group_row = next(row for row in results if row["image_name"] == "VID1_E7_t")
    single_row = next(row for row in results if row["image_name"] == "random_image.png")

    assert group_row["num_cells"] == 20
    assert abs(group_row["plate_coverage_percent"] - 0.5) < 1e-6
    assert single_row["num_cells"] == 10
    assert abs(single_row["plate_coverage_percent"] - 0.4) < 1e-6
