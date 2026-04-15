import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TIME_COLS = ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']
IGNORE_COLS = {'Ave'}
MISSING_VALUE = -99.0


def base_station_id(station_id: str) -> str:
    """Return canonical station id by removing suffix like '-01'/'-02'."""
    return str(station_id).strip().split('-')[0]


def parse_degree_minute(value) -> float:
    """Convert strings like 'N31°56.8\'' to decimal degrees."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace(' ', '')
    if not text:
        return np.nan

    direction = None
    if text[0] in {'N', 'S', 'E', 'W'}:
        direction = text[0]
        text = text[1:]

    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not nums:
        return np.nan

    degree = float(nums[0])
    minute = float(nums[1]) if len(nums) > 1 else 0.0

    decimal = abs(degree) + minute / 60.0
    if degree < 0:
        decimal = -decimal

    if direction in {'S', 'W'}:
        decimal = -abs(decimal)
    elif direction in {'N', 'E'}:
        decimal = abs(decimal)

    return decimal


def build_merge_priority(cols: List[str], base_id: str) -> List[str]:
    """Primary station first, then supplementary stations by suffix order."""
    cols = [c for c in cols if c.startswith(base_id)]

    primary = [c for c in cols if c == base_id]
    supplementary = [c for c in cols if c != base_id]

    def suffix_key(station: str):
        suffix = station.split('-', 1)[1] if '-' in station else ''
        number = int(suffix) if suffix.isdigit() else 9999
        return number, suffix

    supplementary = sorted(supplementary, key=suffix_key)
    return primary + supplementary


def convert_soil_txt_to_csv(txt_path: str, output_csv: str, report_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """Convert SM_NQ TXT to CSV and merge supplementary columns into base station columns."""
    print('Reading soil moisture TXT...')
    raw_df = pd.read_csv(txt_path, sep=r'\s+', engine='python')
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    sensor_cols = [c for c in raw_df.columns if c not in TIME_COLS and c not in IGNORE_COLS]

    base_to_cols: Dict[str, List[str]] = {}
    for col in sensor_cols:
        base = base_station_id(col)
        base_to_cols.setdefault(base, []).append(col)

    merged_df = raw_df[TIME_COLS].copy()
    report_rows = []

    merged_sensor_cols = sorted(base_to_cols.keys())
    for base in merged_sensor_cols:
        candidates = build_merge_priority(base_to_cols[base], base)
        primary = candidates[0]

        merged_values = raw_df[primary].astype(float).copy()
        missing_before = int((merged_values == MISSING_VALUE).sum())
        fill_count_total = 0

        for supplement in candidates[1:]:
            supplement_values = raw_df[supplement].astype(float)
            fill_mask = (merged_values == MISSING_VALUE) & (supplement_values != MISSING_VALUE)
            fill_count = int(fill_mask.sum())
            if fill_count > 0:
                merged_values.loc[fill_mask] = supplement_values.loc[fill_mask]
                fill_count_total += fill_count

        missing_after = int((merged_values == MISSING_VALUE).sum())
        merged_df[base] = merged_values

        report_rows.append({
            'station_id': base,
            'primary_col': primary,
            'supplementary_cols': ';'.join(candidates[1:]),
            'missing_before': missing_before,
            'filled_from_supplement': fill_count_total,
            'missing_after': missing_after,
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    pd.DataFrame(report_rows).to_csv(report_csv, index=False, encoding='utf-8')

    print(f'Soil moisture CSV generated: {output_csv}')
    print(f'  Raw station columns: {len(sensor_cols)} -> merged stations: {len(merged_sensor_cols)}')
    print(f'  Merge report: {report_csv}')

    return merged_df, merged_sensor_cols


def convert_station_xlsx_to_csv(
    xlsx_path: str,
    output_csv: str,
    valid_station_ids: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert station xlsx to CSV and merge supplementary station rows."""
    print('Reading station XLSX...')
    stations = pd.read_excel(xlsx_path)
    stations.columns = [str(c).strip() for c in stations.columns]

    column_map = {
        'Site': 'station_id',
        'Latitude(degree-minute)': 'lat',
        'Longitude(degree-minute)': 'lon',
        'Elevation(meter)': 'elevation',
    }
    stations = stations.rename(columns=column_map)

    needed_cols = [c for c in ['station_id', 'lon', 'lat', 'elevation'] if c in stations.columns]
    stations = stations[needed_cols].copy()

    stations['station_id'] = stations['station_id'].astype(str).str.strip()
    stations['base_station_id'] = stations['station_id'].apply(base_station_id)
    stations['lat'] = stations['lat'].apply(parse_degree_minute)
    stations['lon'] = stations['lon'].apply(parse_degree_minute)

    if 'elevation' in stations.columns:
        stations['elevation'] = (
            stations['elevation']
            .astype(str)
            .str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
            .astype(float)
        )

    merged_rows = []
    missing_station_rows = []
    for station_id in valid_station_ids:
        subset = stations[stations['base_station_id'] == station_id].copy()
        if subset.empty:
            missing_station_rows.append(station_id)
            continue

        primary_row = subset[subset['station_id'] == station_id]
        if primary_row.empty:
            row = subset.iloc[0].copy()
        else:
            row = primary_row.iloc[0].copy()

        for col in ['lon', 'lat', 'elevation']:
            if col in subset.columns and pd.isna(row[col]):
                fallback = subset[col].dropna()
                if not fallback.empty:
                    row[col] = fallback.iloc[0]

        row['station_id'] = station_id
        merged_rows.append(row)

    merged_stations = pd.DataFrame(merged_rows)
    keep_cols = [c for c in ['station_id', 'lon', 'lat', 'elevation'] if c in merged_stations.columns]
    merged_stations = merged_stations[keep_cols].copy()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_stations.to_csv(output_csv, index=False, encoding='utf-8')

    print(f'Station CSV generated: {output_csv}')
    print(f'  Stations with coordinates: {len(merged_stations)}')
    if missing_station_rows:
        print(f'  WARNING: missing coordinates for {len(missing_station_rows)} stations in xlsx')
        print(f'  Missing station ids: {missing_station_rows}')

    return merged_stations, missing_station_rows


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sm_nq_dir = os.path.join(script_dir, 'SM_NQ')

    txt_path = os.path.join(sm_nq_dir, 'SM_NQ-30-minutes_05cm.txt')
    xlsx_path = os.path.join(sm_nq_dir, 'Stations_information_NAQU.xlsx')

    soil_csv = os.path.join(sm_nq_dir, 'SM_NQ-30-minutes_05cm.csv')
    station_csv = os.path.join(sm_nq_dir, 'Stations_information_NAQU.csv')
    merge_report_csv = os.path.join(sm_nq_dir, 'SM_NQ_merge_report.csv')
    missing_station_report_csv = os.path.join(sm_nq_dir, 'SM_NQ_missing_station_report.csv')

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f'TXT file not found: {txt_path}')
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f'XLSX file not found: {xlsx_path}')

    print('=' * 72)
    print('SM_NQ conversion and supplementary station merge')
    print('=' * 72)

    _, station_ids = convert_soil_txt_to_csv(
        txt_path=txt_path,
        output_csv=soil_csv,
        report_csv=merge_report_csv,
    )

    _, missing_stations = convert_station_xlsx_to_csv(
        xlsx_path=xlsx_path,
        output_csv=station_csv,
        valid_station_ids=station_ids,
    )

    if missing_stations:
        soil_df = pd.read_csv(soil_csv)
        drop_cols = [c for c in missing_stations if c in soil_df.columns]
        if drop_cols:
            soil_df = soil_df.drop(columns=drop_cols)
            soil_df.to_csv(soil_csv, index=False, encoding='utf-8')

            merge_df = pd.read_csv(merge_report_csv)
            merge_df = merge_df[~merge_df['station_id'].isin(drop_cols)]
            merge_df.to_csv(merge_report_csv, index=False, encoding='utf-8')

            print(f'  Dropped soil columns not found in station xlsx: {drop_cols}')

    pd.DataFrame({'station_id_missing_in_xlsx': missing_stations}).to_csv(
        missing_station_report_csv,
        index=False,
        encoding='utf-8',
    )

    print('=' * 72)
    print('Done')
    print('=' * 72)
    print('Generated files:')
    print(f'  - {soil_csv}')
    print(f'  - {station_csv}')
    print(f'  - {merge_report_csv}')
    print(f'  - {missing_station_report_csv}')


if __name__ == '__main__':
    main()
