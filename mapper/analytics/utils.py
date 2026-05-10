
from pathlib import Path
from typing import Optional

def parse_optional_int(value: str) -> Optional[int]:
    return int(value) if value not in (None, "", "None") else None

def parse_optional_float(value: str) -> Optional[float]:
    return float(value) if value not in (None, "", "None") else None

def parse_optional_str(value: str) -> Optional[str]:
    return value if value not in (None, "", "None") else None

def parse_optional_path(value: str) -> Optional[Path]:
    return Path(value) if value not in (None, "", "None") else None
