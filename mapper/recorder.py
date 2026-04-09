from pathlib import Path

import csv


class Recorder:
    benchmark_name: str | None = None
    size_x: int
    size_y: int
    sa_algorithm_type: str | None = None
    id: str | None = None

    BASE_PATH = "./benchmarks-results"
    CSV_FILE_NAME = 'data-record.csv'
    CSV_DELIM = ','

    def __init__(self, benchmark_name: str, sa_algorithm_type: str, id: str, size_x: int, size_y: int):
        self.benchmark_name = benchmark_name
        self.sa_algorithm_type = sa_algorithm_type
        self.id = id.replace(":", "-")

        self.size_x = size_x
        self.size_y = size_y

    
    def set_algorithm_type(self, sa_algorithm_type: str):
        self.sa_algorithm_type = sa_algorithm_type

    def get_directory_path(self) -> Path:
        if self.benchmark_name == None or self.sa_algorithm_type == None:
            return self.get_record_csv_file_path()
        return Path(self.BASE_PATH) / self.benchmark_name / f"{self.size_x}x{self.size_y}" / self.sa_algorithm_type  / self.id / self.CSV_FILE_NAME
    
    def get_directory_path_str(self) -> str:
        return str(self.get_directory_path())
    
    def get_record_csv_file_path(self) -> Path:
        return Path(self.BASE_PATH) / self.CSV_FILE_NAME
    
    def record_run(
            self,
            **columns: list[float]
        ):
        """
        We dump the run information inside a file given by directory path
        """
        location = self.get_directory_path()
        location.parent.mkdir(parents=True, exist_ok=True)

        with open(location, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", *columns.keys()])
            writer.writerows((i, *row) for i, row in enumerate(zip(*columns.values())))
