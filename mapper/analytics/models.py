from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Self

from analytics.utils import parse_optional_int, parse_optional_float, parse_optional_path, parse_optional_str

# these are more or less timeouts I put for different benchmarks
EXACT_TIMEOUTS = [500.0, 600.0, 1800.0, 3000.0, 3600.0, 4000.0]
MARGIN_SECONDS = 0.5

# reach row from any CSV can by described by
@dataclass
class Row():
    # additional computed fields
    is_timeout: bool

    # str of syntax [algorithm name]
    # where [algorithm name] is a string
    algorithm: str

    dfg_nodes: int
    size_x: int
    size_y: int

    # str of syntax [algorithm-component]_[algorithm-component]_...
    # where [algorithm-component] is uppercase ASCII string with '-' between words
    sa_algorithm_type: str

    time_seconds: float
    cost: int
    start_configuration_cost: Optional[int]
    iterations: int
    items_each_iteration: Optional[int]
    correctly_positioned_nodes: Optional[int]
    incorrectly_positioned_nodes: Optional[int]

    # str of syntax [[start]->[destination] - [start]->[destination] - ... ]
    # where [start] and [destination] are int
    # some may be malformed
    incorrect_source_destination_nodes: Optional[str]

    # str of syntax [[start]->[destination]: [cost] - [start]->[destination]: [cost] - ... ]
    # where [start], [destination] and [cost] are int
    # some may be malformed
    incorrect_node_cost: Optional[str]

    seed_start_configuration: Optional[int]
    seed_algorithm_run: Optional[int]

    # of type YYYY-MM-DDTHH-MM-SS
    id: datetime

    # additional field that is present if the CSV is archived, is a folder that is part of the path
    # str of syntax DD-MM-YY
    archived_data_prefix: Optional[str]

    # str of type Path
    plot_files_path: Optional[Path]
    # older version have as plot_files_path
    cost_temperature_plot_path: Optional[Path]
    configuration_plot_path: Optional[Path]
    # either one of plot_files_path and the couple cost_temperature_plot_path and configuration_plot_path is set for an instance
    # if it is the latter couple to be set, it may have or may not have configuration_plot_path set
    # we use optional in both as one can be missed

    average_neighbor_sol_time_item: Optional[float]
    average_cost_space_sol_time_item: Optional[float]
    average_sol_check_items_routine_time: Optional[float]
    average_temp_routine_time: Optional[float]
    average_running_time: Optional[float]

    @staticmethod
    def parse_row(row: dict, archived_data_prefix: Optional[str]) -> Self:
        time_seconds = float(row["time_seconds"])
        cost = int(row["cost"])

        # # if we dont find a solution with cost == 0, and the benchmark is saved we have then timed out
        # # for sure
        is_timeout = cost != 0

        return Row(
            # computed data
            is_timeout=is_timeout,

            # CSV data
            algorithm=row["algorithm"],

            dfg_nodes=int(row["dfg_nodes"]),
            size_x=int(row["size_x"]),
            size_y=int(row["size_y"]),

            sa_algorithm_type=row["sa_algorithm_type"],

            time_seconds=time_seconds,
            cost=cost,
            start_configuration_cost=parse_optional_int(row.get("start_configuration_cost", "")),
            iterations=int(row["iterations"]),
            items_each_iteration=parse_optional_int(row.get("items_each_iteration", "")),
            correctly_positioned_nodes=parse_optional_int(row.get("correctly_positioned_nodes", "")),
            incorrectly_positioned_nodes=parse_optional_int(row.get("incorrectly_positioned_nodes", "")),

            incorrect_source_destination_nodes=parse_optional_str(row.get("incorrect_source_destination_nodes", "")),

            incorrect_node_cost=parse_optional_str(row.get("incorrect_node_cost", "")),

            seed_start_configuration=parse_optional_int(row.get("seed_start_configuration", "")),
            seed_algorithm_run=parse_optional_int(row.get("seed_algorithm_run", "")),

            id=datetime.strptime(row["id"], "%Y-%m-%dT%H-%M-%S"),

            archived_data_prefix=archived_data_prefix,

            plot_files_path=parse_optional_path(row.get("plot_files_path", "")),
            cost_temperature_plot_path=parse_optional_path(row.get("cost_temperature_plot_path", "")),
            configuration_plot_path=parse_optional_path(row.get("configuration_plot_path", "")),

            average_neighbor_sol_time_item=parse_optional_float(row.get("average_neighbor_sol_time_item", "")),
            average_cost_space_sol_time_item=parse_optional_float(row.get("average_cost_space_sol_time_item", "")),
            average_sol_check_items_routine_time=parse_optional_float(row.get("average_sol_check_items_routine_time", "")),
            average_temp_routine_time=parse_optional_float(row.get("average_temp_routine_time", "")),
            average_running_time=parse_optional_float(row.get("average_running_time", "")),
        )
    

# Analysis rows to save in a CSV or plot
@dataclass
class AnalyticsColumn():
    sa_algorithm_type: str

    algorithm_mean_timeout: dict[str, bool | None]
    algorithm_median_timeout: dict[str, bool | None]

    algorithm_mean_time: dict[str, float | None]
    algorithm_median_time: dict[str, float | None]

    algorithm_mean_time_norm: dict[str, float | None]
    algorithm_median_time_norm: dict[str, float | None]

    algorithm_mean_iteration: dict[str, float | None]
    algorithm_median_iteration: dict[str, float | None]

    def __to_representative__(
            self,
            macro_group: str,
            id: str = ""):
        return AnalyticsRepresentativesColumn(
            self.sa_algorithm_type, 
            macro_group,
            self.algorithm_mean_timeout,
            self.algorithm_median_timeout,
            self.algorithm_mean_time,
            self.algorithm_median_time,
            self.algorithm_mean_time_norm,
            self.algorithm_median_time_norm,
            self.algorithm_mean_iteration,
            self.algorithm_median_iteration,
            id=id)

@dataclass
class AnalyticsRepresentativesColumn():
    sa_algorithm_type: str
    macro_group: str

    algorithm_mean_timeout: dict[str, bool | None]
    algorithm_median_timeout: dict[str, bool | None]

    algorithm_mean_time: dict[str, float | None]
    algorithm_median_time: dict[str, float | None]

    algorithm_mean_time_norm: dict[str, float | None]
    algorithm_median_time_norm: dict[str, float | None]

    algorithm_mean_time_norm_by_group: dict[str, float | None] = field(default_factory=dict, kw_only=True)
    algorithm_median_time_norm_by_group: dict[str, float | None] = field(default_factory=dict, kw_only=True)

    algorithm_mean_iteration: dict[str, float | None]
    algorithm_median_iteration: dict[str, float | None]

    id: str = ""
