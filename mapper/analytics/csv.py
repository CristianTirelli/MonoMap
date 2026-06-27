

import csv
from pathlib import Path
from typing import Optional
import re

from analytics.models import AnalyticsColumn, AnalyticsRepresentativesColumn, Row
from utils.floats import format_possible_float_number

DD_MM_YY_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{2}$")

def load_csv(csv_path: Path, archived_data_prefix: Optional[str]) -> list[Row]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row: Row = Row.parse_row(row, archived_data_prefix)

            # filter for faulty data when i still used clock time
            delta_clock_time = parsed_row.time_seconds - 4000
            if 0 < delta_clock_time and 1 <= delta_clock_time:
                continue
            
            rows.append(parsed_row)
    return rows


def load_all_rows(CSV_BASE_PATH: Path, CSV_NAME: Path) -> list[Row]:
    all_rows: list[Row] = []

    print(f"Loading CSVs from base path {CSV_BASE_PATH}")

    # Check root CSV
    root_csv = CSV_BASE_PATH / CSV_NAME
    if root_csv.exists():
        all_rows.extend(load_csv(root_csv, archived_data_prefix=None))

    # Check DD-MM-YY subfolders
    for subfolder in CSV_BASE_PATH.iterdir():
        if subfolder.is_dir() and DD_MM_YY_PATTERN.match(subfolder.name):
            subfolder_csv = subfolder / CSV_NAME
            if subfolder_csv.exists():
                all_rows.extend(load_csv(subfolder_csv, archived_data_prefix=subfolder.name))
    return all_rows


def write_analytics_csv(BASE_PATH: Path, algorithm_names: list[str], analytics_column: list[AnalyticsColumn] | list[AnalyticsRepresentativesColumn], id: str = None) -> None:
    if len(analytics_column) == 0:
        print(f"No columns to save to {str(BASE_PATH)}")
        return
    
    is_representatives: bool = False
    if isinstance(analytics_column[0], AnalyticsRepresentativesColumn):
        is_representatives = True

    BASE_PATH.mkdir(parents=True, exist_ok=True)
    file_name = f"analytics{"-representatives" if is_representatives else ""}{"" if not id else f"-{id}"}.csv"
    output_path = BASE_PATH / file_name

    print(f"Writing {file_name} at {str(output_path)}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        representatives_header: list[str] | None
        if is_representatives:
            representatives_header = ["macro-group"] + [c.macro_group for c in analytics_column]
            writer.writerow(representatives_header)

        header = ["algorithm"] + [c.sa_algorithm_type for c in analytics_column]

        writer.writerow(header)
        for algo_name in algorithm_names:
            row = [algo_name] + [format_possible_float_number(c.algorithm_mean_time.get(algo_name, "")) for c in analytics_column]
            for i in range(len(row)):
                if row[i] == None:
                    row[i] = "-"

            writer.writerow(row)
            
        writer.writerow([])

        if is_representatives:
            writer.writerow(representatives_header)

        writer.writerow(header)
        for algo_name in algorithm_names:
            row = [algo_name] + [format_possible_float_number(c.algorithm_median_time.get(algo_name, "")) for c in analytics_column]
            for i in range(len(row)):
                if row[i] == None:
                    row[i] = "-"

            writer.writerow(row)
