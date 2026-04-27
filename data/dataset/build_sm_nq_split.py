#!/usr/bin/env python3
"""Build reproducible SM_NQ train/test/holdout split artifacts.

Outputs:
- Stations_information_NAQU_subset.csv
- SM_NQ-30-minutes_05cm_subset.csv
- test_nodes.npy
- holdout_nodes.npy
- split_summary.json
"""

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd


TIME_COLS = ["yyyy", "mm", "dd", "HH", "MM", "SS"]


def parse_station_values(raw: str) -> List[str]:
    if not raw:
        return []
    values = []
    for token in raw.replace("\n", ",").split(","):
        item = token.strip()
        if item:
            values.append(item)
    return values


def load_station_list(file_path: str) -> List[str]:
    if not file_path:
        return []
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot find station list file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return parse_station_values(f.read())


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def resolve_station_list(inline_values: str, file_path: str) -> List[str]:
    return unique_keep_order(parse_station_values(inline_values) + load_station_list(file_path))


def validate_coordinate_match(
    location_df: pd.DataFrame,
    station_id: str,
    lon: Optional[float],
    lat: Optional[float],
) -> None:
    if lon is None or lat is None:
        return
    row = location_df[location_df["station_id"].astype(str) == station_id]
    if len(row) != 1:
        raise ValueError(f"Cannot uniquely locate station {station_id} in location table")
    lon_ref = float(row.iloc[0]["lon"])
    lat_ref = float(row.iloc[0]["lat"])
    if abs(lon_ref - float(lon)) > 1e-6 or abs(lat_ref - float(lat)) > 1e-6:
        raise ValueError(
            "Provided holdout coordinates mismatch station metadata: "
            f"station={station_id}, expected=({lon_ref}, {lat_ref}), got=({lon}, {lat})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SM_NQ split artifacts with optional holdout station")
    parser.add_argument("--location-path", type=str, default="data/dataset/SM_NQ/Stations_information_NAQU.csv")
    parser.add_argument("--data-path", type=str, default="data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv")
    parser.add_argument("--output-dir", type=str, default="dataset/SM_NQ_split")

    parser.add_argument("--selected-stations", type=str, default="", help="Comma/newline-separated selected stations")
    parser.add_argument("--selected-stations-file", type=str, default="", help="Text file with selected stations")

    parser.add_argument("--test-stations", type=str, default="", help="Comma/newline-separated test stations")
    parser.add_argument("--test-stations-file", type=str, default="", help="Text file with test stations")

    parser.add_argument("--holdout-station-id", type=str, default="", help="Single holdout station id")
    parser.add_argument("--holdout-lon", type=float, default=None, help="Optional holdout station longitude")
    parser.add_argument("--holdout-lat", type=float, default=None, help="Optional holdout station latitude")

    args = parser.parse_args()

    location_df = pd.read_csv(args.location_path)
    data_df = pd.read_csv(args.data_path)

    required_loc_cols = {"station_id", "lon", "lat"}
    if not required_loc_cols.issubset(set(location_df.columns)):
        raise ValueError(f"Location CSV must contain columns: {sorted(required_loc_cols)}")
    for c in TIME_COLS:
        if c not in data_df.columns:
            raise ValueError(f"Data CSV missing required time column: {c}")

    location_ids = set(location_df["station_id"].astype(str).tolist())
    data_station_cols = [c for c in data_df.columns if c not in TIME_COLS]
    data_station_ids = set(data_station_cols)
    available_ids = location_ids.intersection(data_station_ids)

    selected = resolve_station_list(args.selected_stations, args.selected_stations_file)
    if not selected:
        selected = sorted(available_ids)
    else:
        selected = unique_keep_order(selected)

    missing_selected = [s for s in selected if s not in available_ids]
    if missing_selected:
        raise ValueError(f"Selected stations not available in both location/data: {missing_selected}")

    test_stations = resolve_station_list(args.test_stations, args.test_stations_file)
    missing_test = [s for s in test_stations if s not in selected]
    if missing_test:
        raise ValueError(f"Test stations are not in selected stations: {missing_test}")

    holdout_station = args.holdout_station_id.strip()
    if holdout_station:
        if holdout_station not in available_ids:
            raise ValueError(
                f"Holdout station {holdout_station} is not available in both location/data source files"
            )
        if holdout_station in test_stations:
            raise ValueError(f"Holdout station {holdout_station} cannot also be a test station")
        validate_coordinate_match(location_df, holdout_station, args.holdout_lon, args.holdout_lat)
        if holdout_station in selected:
            selected = [s for s in selected if s != holdout_station]
            print(
                f"Strict holdout mode: removed holdout station from selected subset: {holdout_station}"
            )

    train_stations = [s for s in selected if s not in set(test_stations)]
    if len(train_stations) == 0:
        raise ValueError("No training stations left after excluding test/holdout stations")

    selected_sorted = sorted(selected)

    subset_location = (
        location_df[location_df["station_id"].astype(str).isin(selected_sorted)]
        .loc[:, ["station_id", "lon", "lat"]]
        .sort_values(by=["station_id"])
        .reset_index(drop=True)
    )

    subset_data = data_df[TIME_COLS + selected_sorted].copy()

    test_idx = np.array([selected_sorted.index(s) for s in test_stations], dtype=np.int64)
    # Strict zero-participation: holdout node is appended only during holdout inference.
    # Keep this file empty so training/test subsets never include holdout station.
    holdout_idx = np.array([], dtype=np.int64)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    location_out = os.path.join(output_dir, "Stations_information_NAQU_subset.csv")
    data_out = os.path.join(output_dir, "SM_NQ-30-minutes_05cm_subset.csv")
    test_nodes_out = os.path.join(output_dir, "test_nodes.npy")
    holdout_nodes_out = os.path.join(output_dir, "holdout_nodes.npy")
    summary_out = os.path.join(output_dir, "split_summary.json")

    subset_location.to_csv(location_out, index=False)
    subset_data.to_csv(data_out, index=False)
    np.save(test_nodes_out, test_idx)
    np.save(holdout_nodes_out, holdout_idx)

    summary = {
        "location_path": args.location_path,
        "data_path": args.data_path,
        "selected_count": len(selected_sorted),
        "train_count": len(train_stations),
        "test_count": int(len(test_idx)),
        "holdout_count": 1 if holdout_station else 0,
        "selected_stations_sorted": selected_sorted,
        "train_stations": train_stations,
        "test_stations": test_stations,
        "holdout_station": holdout_station,
        "test_indices": test_idx.tolist(),
        "holdout_indices": holdout_idx.tolist(),
        "holdout_mode": "dynamic_append_for_eval" if holdout_station else "none",
        "artifacts": {
            "location_csv": location_out,
            "data_csv": data_out,
            "test_nodes_npy": test_nodes_out,
            "holdout_nodes_npy": holdout_nodes_out,
        },
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Split artifacts generated successfully")
    print(f"  output_dir: {output_dir}")
    holdout_count = 1 if holdout_station else 0
    print(f"  selected/train/test/holdout: {len(selected_sorted)}/{len(train_stations)}/{len(test_idx)}/{holdout_count}")
    print(f"  test_indices: {test_idx.tolist()}")
    print(f"  holdout_indices: {holdout_idx.tolist()}")


if __name__ == "__main__":
    main()
