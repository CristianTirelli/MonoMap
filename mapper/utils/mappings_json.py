import json

# Utility to serialize and deserialize solutions into JSON

def mappings_to_json(
    node_pe: dict[int, int],
    pe_nodes: dict[int, list[int]],
    cost: int = -1
) -> str:
    data = {
        "node_pe": node_pe,
        "pe_nodes": pe_nodes
    }

    if cost != -1:
        data["cost"] = cost

    return json.dumps(data, indent=4, sort_keys=True)

def mappings_to_json_file(
    node_pe: dict[int, int],
    pe_nodes: dict[int, list[int]],
    cost: int = -1,
    path: str | None = None
) -> str:
    json_str = mappings_to_json(node_pe, pe_nodes, cost)

    if path is not None:
        path += '.json'
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str

# TODO, if ever needed
def json_to_mappings(
        path: str
    ) -> tuple[dict[int, int], dict[int, list[int]], int]:
    node_pe: dict[int, int] = {}
    pe_nodes: dict[int, list[int]] = {}
    cost: int = -1

    return (node_pe, pe_nodes, cost)