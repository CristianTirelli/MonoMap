# Backtracking-Enhanced Monomorphism Search for CGRA Mapping

This directory contains the implementations developed as part of the bachelor thesis:

**Backtracking-Enhanced Monomorphism Search for Coarse-Grained Reconfigurable Arrays Mapping**

**Author:** Jora Zeneli  
**Faculty of Informatics, Università della Svizzera italiana (USI)**  
**Bachelor Thesis, 2026**

## Overview

This project builds on the **MonoMap** CGRA mapping framework and investigates
custom backtracking-based algorithms for the spatial mapping phase.

MonoMap decouples the CGRA mapping problem into two main phases:

1. **Temporal search** – an SMT-based scheduler determines when each Data Flow
   Graph (DFG) operation executes.
2. **Spatial search** – the scheduled DFG operations are assigned to Processing
   Elements (PEs) of the CGRA.

The original MonoMap implementation performs the spatial mapping phase using
NetworkX's `GraphMatcher` to search for a graph monomorphism between the
scheduled DFG and the time-expanded CGRA architecture graph.

The work in this project replaces this general-purpose spatial search with a
custom **Depth-First Search (DFS) backtracking framework**.

The custom implementation makes it possible to experiment with different:

- variable-ordering heuristics,
- value-ordering heuristics,
- domain representations,
- constraint-propagation techniques,
- PE-pressure heuristics,
- graph traversal strategies,
- critical-path information,
- and backtracking strategies.

The different `monomap_v*.py` files correspond to experimental mapper versions
developed while investigating these techniques.

---

# Original MonoMap Baseline

## `monomap.py`

`monomap.py` is the original MonoMap implementation and serves as the baseline
for this work.

After generating a valid temporal schedule using the SMT scheduler, MonoMap
constructs a time-expanded architecture graph. Each architecture node represents
a Processing Element at a particular time step.

The scheduled DFG is then spatially mapped onto this architecture graph using
NetworkX's `GraphMatcher` and its subgraph monomorphism search.

This file is kept as the reference implementation against which the custom
backtracking versions can be compared.

---

# Custom Backtracking Framework

The custom mapper implementations replace the `GraphMatcher` spatial search
with a DFS-based backtracking algorithm.

Starting from an empty mapping, the general search procedure is:

1. Select an unmapped DFG node.
2. Compute the architecture nodes to which it may legally be mapped.
3. Order the candidate architecture nodes.
4. Assign the selected DFG node to one candidate.
5. Propagate the consequences of the assignment.
6. Recursively continue the search.
7. If the current partial mapping cannot lead to a complete solution, undo the
   assignment and try another candidate.

The mapper continues until either a complete valid mapping is found or the
search space is exhausted.

## Mapping Constraints

All backtracking versions preserve the fundamental constraints required for a
valid graph monomorphism.

### Injective Mapping

Two DFG nodes cannot be mapped to the same architecture node.

### Time Compatibility

A DFG node can only be mapped to architecture nodes corresponding to the time
step assigned to that operation by the temporal scheduler.

### Connectivity

If two DFG nodes are connected by a dependency, their mapped architecture
nodes must satisfy the connectivity constraints of the CGRA.

---

# Mapper Versions

The following sections describe the different mapper versions developed during
the project.

The versions represent experiments with different variable-ordering,
value-ordering, constraint-propagation, and traversal strategies.

---

## `monomap_v0.py` — Basic DFS Backtracking

The first custom backtracking implementation.

This version introduces the basic recursive DFS/backtracking skeleton that is
used as the foundation for the later implementations.

The mapper selects the first available unassigned DFG node according to the
current time-based ordering and recursively attempts to assign it to compatible
architecture nodes.

The implementation enforces:

- injective mapping,
- time-label compatibility,
- adjacency consistency with already mapped neighbors.

No advanced variable-ordering heuristic is used.

**Main purpose:** establish a simple custom backtracking implementation before
introducing more advanced search heuristics.

---

## `monomap_v0.1.py` — v0 + PE Pressure and Busiest-Time Start

Extension of `v0`.

This version introduces **PE pressure**, stored in a dictionary. PE pressure
tracks how frequently each physical Processing Element is used by the current
mapping.

It also experiments with changing the point from which the search starts.

Instead of beginning from the first scheduled time step, the mapper starts from
the **busiest time step**, defined as the time step containing the largest
number of scheduled DFG nodes.

The motivation is to begin the search from a more constrained part of the
mapping problem.

---

## `monomap_v1.py` — Minimum Remaining Values (MRV)

This version introduces the **Minimum Remaining Values (MRV)** variable-ordering
heuristic.

Instead of simply selecting the first unassigned DFG node, the mapper computes
the currently feasible domain of each unassigned node and selects the node with
the **smallest remaining domain**.

A domain represents the set of architecture nodes to which a DFG node may
currently be mapped.

The intuition behind MRV is to map the most constrained node first. Nodes with
few available placements are more likely to cause a failure, so considering
them early can expose infeasible branches sooner.

This implementation also introduces stronger search mechanisms, including:

- dynamic domain computation,
- degree-based tie-breaking,
- forward checking,
- Least-Constraining Value (LCV) candidate ordering.

---

## `monomap_v1.1.py` — Incremental Domain Storage

This version optimizes the domain handling introduced in `v1`.

Instead of repeatedly recomputing every domain from scratch, the current domains
are stored in a dictionary and updated incrementally as assignments are made.

When a node is assigned:

1. Its domain is restricted to the selected architecture node.
2. The selected architecture node is removed from the domains of other
   unassigned nodes to preserve injectivity.
3. Domains of neighboring DFG nodes are restricted according to architecture
   connectivity.
4. All domain modifications are recorded so they can be undone during
   backtracking.

This implementation therefore maintains the search state incrementally rather
than reconstructing it at each recursive call.

**Main purpose:** reduce the overhead of repeatedly computing domains during the
backtracking search.

---

## `monomap_v1.2.py` — MRV + PE Pressure and Busiest-Time Start

This version extends the MRV-based mapper with the PE-pressure and
busiest-time-step experiments.

It combines:

- dynamic domains,
- MRV node selection,
- PE-pressure tracking,
- and search initialization from the busiest scheduled time step.

The goal is to combine constraint-based node selection with information about
CGRA resource usage and temporal congestion.

---

## `monomap_v1.3.py` and `monomap_v1.4.py` — PE-Pressure Value Ordering

These versions experiment with using **PE pressure** as a value-ordering
heuristic.

Once a DFG node has been selected, candidate architecture nodes are ordered
according to the current pressure of their corresponding physical PE.

Two opposite strategies are investigated:

- prioritizing **less-loaded PEs**, attempting to distribute operations more
  evenly across the CGRA;
- prioritizing **more-loaded PEs**, attempting to concentrate assignments on
  PEs that are already heavily used.

These experiments investigate whether spreading or concentrating PE usage can
guide the backtracking search toward valid mappings more efficiently.

---

## `monomap_v1.5.py` — MRV + Descendant-Aware Ordering

This version introduces a more detailed variable-ordering strategy.

When choosing the next DFG node, candidates are prioritized according to:

1. **smallest feasible domain**,
2. **largest number of descendants**,
3. **highest graph degree**,
4. **node ID** as the final deterministic tie-break.

The descendant count represents how many other DFG nodes can be reached from a
node through directed paths.

The motivation is that a node with many descendants can influence a large
portion of the remaining computation. Mapping such nodes earlier may therefore
provide useful constraints for later decisions.

This version combines information about:

- the current search state through MRV,
- the directed structure of the DFG through descendants,
- and local connectivity through graph degree.

---

## `monomap_v1.4.1.py` — Descendant-Aware Ordering + PE Pressure

This version combines the node-selection strategy used by the descendant-aware
mapper with PE-pressure-based candidate ordering.

Node selection considers:

1. smallest feasible domain,
2. largest number of descendants,
3. highest graph degree,
4. node ID.

Candidate architecture nodes are then ordered using PE pressure, prioritizing
the more heavily loaded PEs.

This version therefore combines a structural variable-ordering heuristic with a
resource-aware value-ordering heuristic.

---

## `monomap_v2.py` — Highest-Degree Node Ordering

This version changes the variable-selection strategy from MRV to a
**highest-degree heuristic**.

The next DFG node is selected according to its graph degree.

A node with a high degree participates in many dependency constraints.
Assigning highly connected nodes earlier may therefore restrict the remaining
search space more quickly and expose conflicts earlier.

**Main purpose:** evaluate whether graph connectivity alone can provide an
effective variable-ordering strategy.

---

## `monomap_v2.1.py` — Degree Ordering + PE Pressure and Busiest-Time Start

Extension of `v2`.

This version combines:

- highest-degree node selection,
- PE-pressure tracking,
- and initialization from the busiest scheduled time step.

It evaluates whether the degree-based mapper benefits from the same
resource-aware and time-step-based search guidance explored in the other
versions.

---

## `monomap_v3.py` — Directed Parent/Child Traversal

This version experiments with explicitly using the **direction of the DFG** to
guide the mapping order.

Instead of selecting nodes using only generic graph properties, the search
traverses parent/child relationships in the directed DFG.

The search attempts to continue through related operations by moving through
children and parents before selecting a new unrelated node.

Candidate placement can also take the locations of already mapped parents into
account.

**Main purpose:** investigate whether following the dependency structure of the
DFG provides better search guidance than generic node ordering.

---

## `monomap_v4.py` — Conflict-Directed Backjumping

This version extends the MRV-based mapper with **Conflict-Directed Backjumping
(CBJ)**.

Standard chronological backtracking returns to the most recently assigned node
whenever a dead end is reached.

However, the most recent assignment is not necessarily responsible for the
failure.

Conflict-Directed Backjumping tracks a **conflict set** containing the previous
assignments that contributed to a domain becoming infeasible.

When a failure occurs, the search can use this information to jump directly
back to a relevant decision instead of always moving backward one recursion
level at a time.

This version includes:

- dynamic domains,
- MRV node selection,
- degree-based tie-breaking,
- forward checking,
- LCV candidate ordering,
- conflict-set construction,
- conflict-directed backjumping.

**Main purpose:** reduce unnecessary chronological backtracking by identifying
which previous assignments actually caused a failure.

---

## `monomap_v5.py` — Topological Node Ordering

This version selects DFG nodes according to a **topological ordering**.

Rather than dynamically selecting the next node using MRV or graph degree, the
mapper follows an ordering derived from the dependency structure of the DFG.

This provides a deterministic dependency-aware alternative to MRV and
degree-based node selection.

**Main purpose:** investigate whether following the natural dependency order of
the computation improves the spatial search.

---

## `monomap_v6.py` — Immediate-Child-Guided Ordering

This version experiments with using the **immediate children** of DFG nodes to
guide the mapping order.

The mapper uses local parent/child dependency information to influence which
node is explored next.

Unlike descendant-based ordering, which considers all reachable downstream
nodes, this strategy focuses on immediate child relationships.

**Main purpose:** investigate whether local dependency structure provides useful
guidance for the backtracking search.

---

## `monomap_v6.1.py` — MRV + Immediate-Child Guidance

This version combines the MRV-based strategy with immediate-child information.

MRV remains responsible for identifying constrained nodes, while child-related
information is used as additional guidance when ordering nodes.

The implementation therefore combines:

- dynamic domains,
- MRV,
- immediate-child information,
- descendant information,
- degree-based information,
- forward checking,
- LCV candidate ordering.

**Main purpose:** determine whether dependency-aware child information can
improve the standard MRV search.

---

## `monomap_v6.2.py` — Child-Threshold Descendant Expansion

This version extends the child-guided approach with a **child threshold**.

The mapper considers how many immediate children a node has. When a node
exceeds the configured child threshold, its downstream descendants are given
additional priority during the search.

The goal is to identify nodes that influence a relatively large downstream
region of the DFG and map that region earlier.

**Main purpose:** combine local child information with broader descendant
expansion.

---

## `monomap_v7.py` — Critical-Path-First MRV

This version extends the MRV-based search with **critical-path information**.

The mapper identifies the critical path of the DFG using longest-path
information after handling the graph's back edges.

Nodes belonging to the critical path receive additional priority during node
selection.

The motivation is that critical-path operations belong to one of the most
dependency-constrained parts of the computation. Mapping these operations
earlier may constrain the remaining search more effectively.

This version combines the standard backtracking mechanisms with
critical-path-guided variable ordering.

---

# Statistics Versions

Additional versions of some mappers contain instrumentation for collecting
statistics during the backtracking search.

These include files such as:

- `monomap_v1stats.py`
- `monomap_v1.2stats.py`
- `monomap_v1.4stats.py`
- `monomap_v1.5stats.py`

These implementations use the corresponding mapper strategy while additionally
collecting information about the behavior of the search.

They are intended for experimental analysis and are not separate mapping
algorithms.

---

# Summary of Mapper Versions

| Version | Main Search Strategy | Main Difference |
|---|---|---|
| `monomap.py` | NetworkX GraphMatcher | Original MonoMap baseline |
| `v0` | Basic DFS | First custom backtracking implementation |
| `v0.1` | Basic DFS + PE pressure | Busiest-time start and PE-pressure tracking |
| `v1` | MRV | Smallest feasible domain first |
| `v1.1` | MRV + incremental domains | Domains stored and updated incrementally |
| `v1.2` | MRV + PE pressure | Busiest-time start and PE-pressure tracking |
| `v1.3` / `v1.4` | MRV + PE-pressure ordering | Experiments with low/high PE pressure |
| `v1.5` | MRV + descendants + degree | Composite variable-ordering heuristic |
| `v1.4.1` | Descendant-aware + PE pressure | Composite node ordering with PE-pressure value ordering |
| `v2` | Highest degree | Most connected DFG node first |
| `v2.1` | Highest degree + PE pressure | Degree ordering with busiest-time/pressure experiment |
| `v3` | Directed traversal | Parent/child-guided exploration |
| `v4` | MRV + CBJ | Conflict-directed backjumping |
| `v5` | Topological order | Dependency-based fixed ordering |
| `v6` | Child-guided | Immediate-child-based ordering |
| `v6.1` | MRV + child guidance | MRV combined with child information |
| `v6.2` | Child-threshold ordering | Descendant expansion based on number of children |
| `v7` | Critical-path + MRV | Prioritizes critical-path operations |

---

# Running the Benchmark Suite

The different mapper implementations can be evaluated using
`run_benchmarks.py`.

The general command is:

```bash
python3 run_benchmarks.py \
    -script <mapper_file.py> \
    -x <CGRA_X> \
    -y <CGRA_Y> \
    -d <TOPOLOGY_DEGREE> \
    -o <output_file.csv>
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `-script` | Mapper implementation that should be executed |
| `-x` | X dimension / number of rows of the CGRA |
| `-y` | Y dimension / number of columns of the CGRA |
| `-d` | Topology degree used for the CGRA configuration |
| `-o` | Name of the CSV file in which the benchmark results are stored |

## `-script`

The `-script` argument selects which mapper implementation should be evaluated.

For example:

```bash
-script monomap_v1.py
```

runs the MRV-based mapper, while:

```bash
-script monomap_v4.py
```

runs the conflict-directed backjumping implementation.

Any of the mapper versions can be evaluated by changing this argument.

## `-x` and `-y`

These arguments define the dimensions of the target CGRA.

For example:

```bash
-x 5 -y 5
```

selects a **5 × 5 CGRA**, containing 25 physical Processing Elements.

Similarly:

```bash
-x 20 -y 20
```

selects a **20 × 20 CGRA**, containing 400 physical Processing Elements.

## `-d`

The `-d` argument specifies the topology degree used for the selected CGRA
configuration.

The value must be compatible with the topology configuration expected by the
mapper.

## `-o`

The `-o` argument specifies the CSV file in which the benchmark results should
be stored.

For example:

```bash
-o results_v1_5x5.csv
```

stores the results in `results_v1_5x5.csv`.

Using descriptive output names is recommended when comparing multiple mapper
versions and CGRA sizes.

---

# Experimental Configurations

The thesis evaluates the mapping approaches across different CGRA sizes:

| CGRA | Number of PEs |
|---|---:|
| 2 × 2 | 4 |
| 5 × 5 | 25 |
| 10 × 10 | 100 |
| 20 × 20 | 400 |

The benchmark suite contains kernels from several application domains,
including:

- cryptography,
- signal processing,
- image processing,
- numerical and scientific computing,
- pattern matching.

A timeout of **4000 seconds** was used for each benchmark execution during the
experimental evaluation.

---

# Development Overview

The mapper versions represent an experimental progression of the custom spatial
search.

The general development path can be summarized as:

```text
Original MonoMap
    |
    +-- NetworkX GraphMatcher spatial search
    |
    v
v0: Basic custom DFS backtracking
    |
    +-- v0.1: PE pressure + busiest-time start
    |
    v
v1: MRV + dynamic domains + forward checking + LCV
    |
    +-- v1.1: Incrementally maintained domains
    |
    +-- v1.2: PE pressure + busiest-time start
    |
    +-- v1.3 / v1.4: PE-pressure value ordering
    |
    +-- v1.5: MRV + descendants + degree
    |
    +-- v1.4.1: Descendant-aware ordering + PE pressure
    |
    +-- v4: Conflict-directed backjumping
    |
    +-- v6.1: MRV + immediate-child guidance
    |
    +-- v7: Critical-path-guided MRV

v2: Highest-degree ordering
    |
    +-- v2.1: PE pressure + busiest-time start

v3: Directed parent/child traversal

v5: Topological ordering

v6: Immediate-child-guided ordering
    |
    +-- v6.2: Child-threshold descendant expansion
```

The purpose of preserving these versions is to make it possible to study how
different search decisions affect the behavior of the spatial mapper.

The experiments focus particularly on the effect of:

- constraint propagation,
- variable ordering,
- value ordering,
- dependency-aware traversal,
- resource-aware PE selection,
- and alternative backtracking strategies.