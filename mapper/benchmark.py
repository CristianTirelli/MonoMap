from pathlib import Path

import fcntl

class Benchmark:
    benchmark_name: str | None = None
    sa_algorithm_type: str | None = None
    id: str | None = None

    size_x: int | None = None
    size_y: int | None = None
    dfg_nodes: int | None = None

    BASE_PATH = "./benchmarks-results"
    CSV_FILE_NAME = 'data-benchmarks.csv'
    CSV_DELIM = ','

    def __init__(self, benchmark_name: str, sa_algorithm_type: str, id: str, size_x: int, size_y: int):
        self.benchmark_name = benchmark_name
        self.sa_algorithm_type = sa_algorithm_type
        self.id = id.replace(":", "-")

        self.size_x = size_x
        self.size_y = size_y

    
    def set_algorithm_type(self, sa_algorithm_type: str):
        self.sa_algorithm_type = sa_algorithm_type

    def set_dfg_nodes(self, dfg_nodes: int):
        self.dfg_nodes = dfg_nodes

    def get_directory_path(self) -> Path:
        return Path(self.BASE_PATH) / self.benchmark_name / f"{self.size_x}x{self.size_y}" / self.sa_algorithm_type
    
    def get_directory_path_str(self) -> str:
        return str(self.get_directory_path())
    
    def get_directory_path_no_size_str(self) -> str:
        return str(Path(self.BASE_PATH) / self.benchmark_name)
    
    def get_benchmarks_csv_file_path(self) -> Path:
        return Path(self.BASE_PATH) / self.CSV_FILE_NAME
    
    def save_results(
            self,
            time: float,
            cost: int,
            iterations: int,
            cost_temperature_plot_path: str,
            configuration_plot_path: str,
            correctly_positioned_nodes: int,
            incorrectly_positioned_nodes: int,
            incorrect_source_destination_nodes: dict[int, int],
            incorrect_node_cost: dict[int, int]
        ):
        """
        We save the time result in a csv file at `save_to` with the correct `image_id`.

        If not present we create one

        If path or id is not provided we throw an `AssertionError`

        CSV files have the following structure:

        ```
        algorithm, size_x, size_y, sa_algorithm_type, time_seconds, cost, iterations, id
        x,y,z, ...
        ...
        ```

        where:

        - algorithm is the name of the benchmarked algorithm
        - size_x the number of columns of the CGRA
        - size_y the number of rows of the CGRA
        - sa_algorithm_type the name of the Simulated Annealing module the algorithm is imported from
        - time_seconds the time taken for the algorithm to complete in seconds (if 4000 it timed out)
        - cost the cost at the end of the algorithm
        - iterations the number of iterations done
        - id the id of the run, which is the time in seconds in ISO format
        """
        if not self.benchmark_name or not self.sa_algorithm_type or not self.id:
            raise AssertionError("Missing location information")
        
        # path to data file
        location = self.get_benchmarks_csv_file_path()

        # create folders, check if file is present and if it is empty
        location.parent.mkdir(parents=True, exist_ok=True)
        file_exists = location.exists()
        is_empty = file_exists and location.stat().st_size == 0
        needs_header = not file_exists or is_empty

        # parse to valid string

        incorrect_source_destination_nodes_strings: list[str] = []
        for key, value in incorrect_source_destination_nodes.items():
            incorrect_source_destination_nodes_strings.append(f"{key}->{value}")
        str_incorrect_source_destination_nodes: str = f"[{" - ".join(incorrect_source_destination_nodes_strings)}]"

        str_incorrect_node_cost_strings: list[str] = []
        for key, value in incorrect_node_cost.items():
            str_incorrect_node_cost_strings.append(f"{key}->{value}")
        str_incorrect_node_cost: str = f"[{": ".join(str_incorrect_node_cost_strings)}]"

        # "a" creates if not present
        with location.open(mode="a", encoding="utf-8", newline="") as csv_file:
            fcntl.flock(csv_file, fcntl.LOCK_EX)
            if needs_header:
               # add headers if file is new or empty
                print(self.CSV_DELIM.join([
                    "algorithm",
                    "dfg_nodes",
                    "size_x",
                    "size_y",
                    "sa_algorithm_type",
                    "time_seconds",
                    "cost",
                    "iterations",
                    "correctly_positioned_nodes",
                    "incorrectly_positioned_nodes",
                    "incorrect_source_destination_nodes",
                    "incorrect_node_cost",
                    "id",
                    "cost_temperature_plot_path",
                    "configuration_plot_path"
                ]), file=csv_file, flush=True)
            # then add data row
            print(self.CSV_DELIM.join([
                self.benchmark_name,
                str(self.dfg_nodes),
                str(self.size_x),
                str(self.size_y),
                self.sa_algorithm_type,
                str(time),
                str(cost),
                str(iterations),
                str(correctly_positioned_nodes),
                str(incorrectly_positioned_nodes),
                str_incorrect_source_destination_nodes,
                str_incorrect_node_cost,
                self.id,
                str(Path.cwd() / cost_temperature_plot_path),
                str(Path.cwd() / configuration_plot_path)
                ]), file=csv_file, flush=True)
            fcntl.flock(csv_file, fcntl.LOCK_UN)
