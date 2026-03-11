# Bachelor Project

Collection of all bachelor project related notes and logs I am taking

## Log

### Setup

I started by trying to do the setup. Python setup was flawless but LLVM
compilation returned an error for the file `CGRAExtract.cpp`, which looking at
the github repository of the forked LLVM project it is a custom file implemented
by Cristian. The error returned was:
`CGRAExtract.cpp:7:10: fatal error: 'bits/stdc++.h' file not found`. I did some
research and the header is just (supposedly) a quicker way to import majority of
standard libraries. The file ships for Linux distributions with GCC compiler, on
macOS it is usually used Clang, which uses Apple’s libc++ backend, which does
not include this header. To resolve the error I remved the header and added
standard libraries imports. I have modified CGRAExtract.cpp.

Now that LLVM compiles I opened the python environemnt and run
`./monolang -f benchmarks/sha2/sha.c`, which returns an error when compiling the
C source code: `sha.c:1:10: fatal error: 'stdio.h' file not found` it does not
seem to be able to locate standard libraries. The step that breaks is
`./llvm-project/build/bin/clang -O3 -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o extracted.ll ./benchmarks/sha2/sha.c`
The fact is that in linux (Ubuntu) libraries are probably in a path in which
they can be found easily by clangd LLVM, on macOS that's probably not the case.
To fix it we need to add the path to the standard libraries in macOS either at
compilation or when running the command as command line parameter. To do so we
use `xcrun` that returns the path of where the XCode SDK libraries are located.
Added bash conditional statement to check the OS, and add libraries linkage if
on darwin (macOS).

I also get a warning when running the next step, i.e.
`./llvm-project/build/bin/opt -disable-output extracted.ll -passes=cgra-extract`
which says:
`./llvm-project/build/bin/opt: WARNING: failed to create target machine for 'arm64-apple-macosx16.0.0': unable to get target for 'arm64-apple-macosx16.0.0', see --version and --triple.`.
Looking in the directory I can see that it actually run correctly I suppose,
even with the warning. Which probably suggest that is not my architecture in the
wrong but that the LLVM clangd was not compiled with the ARM64 backend, which
implies that it can't generate binary to build the executable for the `.ll`
"target" architecture. Since our job does not concern running the code on any
architecture we can also not worry about the warning as we just need it to run
the custom script to build the DFG. The command line option
`-passes=cgra-extract` is probably the input that makes the compiler execute the
custom LLVM code to generate the DFG.

After the python monomap script completed the returned output is:

```
16 CGRA size
acc1/acc_edges
4 x 4
Parsing done!
NOTE: The architecture time is not included in the final compilation time.
RecII is computed with recursive_simple_cycles from networkX.
Sometimes it doesn't provide the correct lowerbound.
To manually set the II , use the -i option.
II = max(RecII, ResII) = max(10, 2)= 10
#nodes:  25
#edges:  33
#maxdegree:  4
Start schedule generation
Scheduling II = 10
16
Time to generate constraints: 0.0742180347442627
Start solving...
MODEL 0
MODEL 1
MODEL 2
MODEL 3
Time to find schedule: 0.07861876487731934
End schedule generation:  0.16545319557189941
Len schedule 10
Schedule
0 [4, 5]
1 [1, 2, 3, 6, 7]
2 [8, 10]
3 [0, 11]
4 [22]
5 [9, 15, 23]
6 [12, 16, 24]
7 [13]
8 [14, 17, 19, 20]
9 [18, 21]
Start architecture graph generation
Time to generate architecture: 0.010675907135009766
Monomorphism search start...
Time for monomorphism search: 0.11566400527954102
Monomorphism found!
Final mapping
Node  5  Mapped on PE  0  at time  0
Node  4  Mapped on PE  1  at time  0
Node  3  Mapped on PE  0  at time  1
Node  2  Mapped on PE  1  at time  1
Node  1  Mapped on PE  2  at time  1
Node  6  Mapped on PE  4  at time  1
Node  7  Mapped on PE  3  at time  1
Node  8  Mapped on PE  0  at time  2
Node  10  Mapped on PE  1  at time  2
Node  0  Mapped on PE  0  at time  3
Node  11  Mapped on PE  1  at time  3
Node  22  Mapped on PE  0  at time  4
Node  23  Mapped on PE  3  at time  5
Node  9  Mapped on PE  0  at time  5
Node  15  Mapped on PE  1  at time  5
Node  12  Mapped on PE  0  at time  6
Node  16  Mapped on PE  1  at time  6
Node  24  Mapped on PE  3  at time  6
Node  13  Mapped on PE  0  at time  7
Node  20  Mapped on PE  2  at time  8
Node  17  Mapped on PE  1  at time  8
Node  14  Mapped on PE  0  at time  8
Node  19  Mapped on PE  5  at time  8
Node  18  Mapped on PE  0  at time  9
Node  21  Mapped on PE  1  at time  9
Total time: 0.28111720085144043
```

I see that `extracted.ll` is the LLVM IR code, and that the `cgra-mono-code-acc1` is
the output of `monomap.py` given graph structure of all `acc*` folders. As
discussed the `.dot` is the format of the DFG produced by the custom LLVM
program that parsed the LLVM IR file with dependencies (loop denoted with
`[color=red]`). I suppose `acc_edges` is just the list of edges in the graph.

### Simulated Annealing Implementation

After making plots for the DFG and the CGRA positioning I started laying out the first version of the simulated annealing algorithm.
We currently enter the function with more data than used.
It relies on both x and y sizes, the schedule, the dfg and the architecture graph `arch`.
We start by generating a random solution and computing its cost.
Then we iterated until we find the solution or the temperature reaches the freezing temperature.
We update the temperature after seeing at least, for now, 50 solutions.
If the found solution is better we accept it right away and if it is worse we accept it by following the equation: `math.exp(- delta_E / T)` where `delta_E` is the difference in cost from the best solution found until now and the solution just found.

Right now we define the cost function as the squared distance from all DFG dependencies that are not connected, i.e. if a dependency is connected (it is satisfied) we do not increase the cost, otherwise if it is not we add a cost proportional to the distance of PEs of the instructions squared to signify that the further it is the worse it is (a dependency that has to hop 3 PEs is much more worse than a dependency that has to hop 2 PEs).

We generate solutions completely at random, we loop through the schedule and for each instant we randomnly select a PE for each instruction at random, discarding the selected PE for the next instruction.

By running it multiple times we find that the solutions never go under a cost of 50, never moving near a solution.
This is probably a matter of choosing solutions.
We are not choosing a neighbour, we are merely going randomly within the solution space.
We need to change the generating function.
It is okay to start with a random solution but then each solution generated from the iteration has to be a "neighbour" solution, but how close should it be? How can we define a neighbour function?
Could keeping lowest x cost edges intact when searching a neighbouring solution work?

After implementing it, it runs up to a solution with a cost of under 40, an improvement on the complete random strategy that yielded a best result of right under 60 of cost.
Still, it is not going to do the job.
I remember Cristian saying that the algorithm could move by computing the inner layer at first and then move outwards and I think that such strategy could work well.
So we rewrite the routine to account for that.
The problem is that layers do not always connect directly, so if we simulate placing for a clock time `t` and  `t + 1` we may not be able to evaluate the cost of such layout as they may not have direct dependencies, and as such any placement can be viewed as correct.
I don't know, what about moving around a random operation at a time by moving it by 1 place?

Moving around a random operation at a time by moving it at random leads us to a valid solution for `sha.c` and a CGRA of 4x4.
The result of first implementation takes 0.28s compared to 0.11s of monomorphic search.
Let's try and refine the code of the search and eliminating all computation that is not part of the algorithm.
It improves to 0.19s.
What about other problem and with different sizes?

| Problem | Size | Monomorphic | SA |
| :--- | :--- | :--- | :--- |
| sha2 | 4x4 | 0.11 | 0.18 |
| sha2 | 3x4 | 0.13 | 0.10 |
| sha2 | 3x3 | 0.039 | 0.05 |
| gsm | 3x3 | 0.0003 | 0.01 |
| bit_count | 3x3 | 0.0001 | 0.0003 |
| reverse_bits | 3x3 | 0.0002 | 0.0000348 |
| sha1 | 3x3 | 0.0005 | 0.019 |
| sqrt | 3x3 | 0.0002 | 0.006 |
| string_search | 3x3 | 0.0003 | 0.014 |

Still is it the best a Simulated Annealing Algorithm can get?
Would it be better if we add a more heuristic move when searching for a neighbour and moving operations around?

The fact is that the last methodology works is because of the neighbour search, moving too much the solution from the current one makes it for a random walk (random solution each time) instead of a search and moving too little prevents it from moving within the search space (high edge keeping percentage).
By increasing the number of best edge dependencies to maintain to search for a neighbour the algorithm gets stuck at a specific cost, indicating that it is probably too restrictive, i.e. it needs more freedom to move within the search space.
At 0.5 (percentage of best edges to consider their operation nodes fixed when searching a neighbouring solution) and lower it floats in the vicinity of a cost of 60 and at 0.6 gets stuck at a cost of over 100 and doesn't seem to be able to find better solutions, in between percentages move around those two cases.

Moving at random one node is great and seems to always find a solution, if available, and at a quite good pace.
Now I don't know if it is a good enough strategy to go and optimize or if we can do better by applying a different search.
From [morpheus](https://github.com/ecolab-nus/Morpher_CGRA_Mapper/blob/stable/include/morpher/mapper/SimulatedAnnealingMapper.h#L172) they update the temperature based on acceptance rate, let's see how it behaves in our system.
Still, I revisited as they decrease temperature the more acceptance rate the current temperature has, I inverted it as I want to "stabilize" the search before moving on, which makes it for a better exploration of the space and solutions vicinity.
The chart that is produced displays a very gentile descent towards zero cost, yet the time it takes to produce a solution greatly increases.
It can't find solutions for CGRA 10x10 and sha2, as it reaches freezing before it.
By decreasing temperture in two steps it finds a solution, it takes 23s.

Now I think we need to improve / refine components:

- Neighbour function
- Cost Function
- Temperature decrease

I think that the algorithm might be able to do mre sophisticated work such as identifying a solution that uses as many distinct PEs as possible, as it could be a good metric as to make the chip last as much as possible, i.e. we do not deteriorate single PE pins by always using them over and over again.

For a 20x20 and sha2 it takes the random node movement with two step temperature about 2 minutes to find a solution (124.25s).

### SA Performances

A fast SA usually has traits:

- Do not compute completely a neighbouring solution, starts from an already constructed one and changes only parts that need to be changed
- Not selecting a random operation but following an heuristic can improve convergence rate, but it can also detrement search chances
- Good management of temperature (tied to the problem)

Could it have:

- tabu list that prevents from choosing bad neighbours that have been found (follows temperature, as it decrease the more restricting it becomes), could save computation.

### SA Neighbour functions

A neighour function can be:

- Move a random operation
- Move x random operations (too many goes outside neighbour)
- Move the highest distance operation (has heuristic)

### Temperature Traversal

Before implementing different searches I want to tie together temperature and the search, not in the fact that the actual problem gives us intuition about temperature coefficients such as the start temperature the decreasing of the temperature or the schedule it follows, that is hard to accomplish and may come later.
What I am actuallly referring to is that the temperature should be aware of the current trajectory, like Cristian said, if you are very near to the solution increase the temperature and keep going a little bit as you might just need some more time to find it.
Right now computing the `particlefilter` problem (with 10x10 sizing) I ran in the problem that the algorithm freezes before it finds a solution, even while it being very near and being in a steady but promising cost descent.
What I want to start to do with the temperature is to tie it to the search, we could keep track of the moving average of the selection of new least cost solutions, and if reaching temperature while having the moving average indicate that we are actually in a descent we can give it more time and make it compute a little bit more.
We start by a simple moving average

$$SMA_K = \frac{1}{k} \sum_{i = n - k + 1}^n p_i$$
$$SMA_{k, next} = SMA_{k, prev} + \frac{1}{k}(p_{n+1} - p_{n-k+1})$$

With a $k$ of maximum 50 elements keeping track of the cost movement.
But, by only having one how can I distinguish downtrends? and with what criterion?
Using only two previous sma points seems to distruptive: you cut out from the search searches that are coming out with good solution less frequently.
I opted to add a second SMA, one that takes much more elements than the other and look for the crossing: a faster SMA that is under the longer SMA indicates that we are trending down.

The implementation works and gets stuck: It works as the temperature gets refilled as we are trending down, but as we do solutions get probably more cluttered with randomness, as the solution space is very large that we get in this loop where we improve very little after a lot of time, just enough to keep it increasing the temperature.

Another fact is that the faster SMA will never cross up the slower sma (and freeze the system until they see the same cost of quite some time), and if the cost doesn't drop it won't get additional temperature, so this is intentional.

What it might help is what we talked about hybridicity: as we increase temperature it may be better to keep seeking heuristic moves that exploit how a solution shoub be constructed instead of random as we aim at improving the local minimum.
Might also be the case that we can add a deterministic refinment procedure such as the 2-OPT and 3-OPT for the TSP to maybe resolve randomness articafts that do not make sense.

This implementation is too volatile and too slow, we don't adress the problem that might arise when at low temperature we get stuck in a local minimum that has no actual solution (possible?).

We move to implementing different searches.

### Implementing Priorities Strategies

I implemented the worst nodes priority with poisson distribution draw strategy and it feels somewhat incosistent at first, I am trying to figure out the lambda that fits bets the algorithm.
I think that it should follow how many operation nodes there are in the problem, so that it scales up and down with each problem.
It may be that the percentage does not work correctly for all kind of problems.

### Implementing Different Searches

The task is to implement different types of searches and benchmark them to see which one performs best.
We should take problems that are hard to find with monomorphism like the timed out problems of the paper's benchmarks.
First we implement strategies.
Strategies that we implemented and that do not seem to find solutions:

- Random Worst Node (`sa_random_worst_node.py`)
- Best Edges (`sa_best_edges.py`)

Strategies that we implemented and that seem to provide good solutions for smaller problems are:

- Random Node (`sa_random_node.py`)
- Random Node with tied temperature (`sa_random_node_ttied.py`)

### Implementation of the Benchmark

After having implemented a few different strategies it appeared clear that we can't get a clear intuition from the few untracked results that we get.
So I started creating a benchmarking script (both in bash and in python) to run the algorithm multiple times adn record each result and picture of each run so that we can get some data on how the algorithm is behaving and to compare different algorithms at once.
The structure is not simple, we have different kind of problems such as sha1, sha2, particlefilter, etc. which in turn have different CGRA sized such as 2x2, 5x5, 10x10, etc. which can be run with different Simulated Annealing strategies such as random node, worst nodes with poisson distruction, etc.
I thought about building a single csv file to contain all of these combinations, the end result should be as follows:

We save the DFG graph for each algorithm in `benchmarks-results/{algorithm}`, where `{algorithm}` is the name of the algorithm we are solving for

We save the CGRA PE Mappings graph for each algorithm we are solving for, CGRA Size and SA algorithm type in `benchmarks-results/{algorithm}/{size_x}-{size_y}/{sa_algorithm_name}-mappings-CGRA-{id}.png`, where:

- `{algorithm}` is the name of the algorithm we are solving for
- `{size_x/y}` are the CGRA x and y size
- `{sa_algorithm_name}` is the name of the Simulated Annealing algorithm type
- `{id}` is the time until seconds of when the benchmark has ben run

We also save the run Cost and iterations graph detailed with temperature and the different data that we use during the search for each algorithm we are solving, CGRA Size and SA algorithm type in `benchmarks-results/{algorithm}/{size_x}-{size_y}/{sa_algorithm_name}-costs-vs-temperature-{id}.png`, where each variable remains the same as for the CGRA mappings file.

This way you can index images results of the different runs easily while looking at the information in the csv file, which is structured as:

Algorithm we solve for, Size x, Size y, Simulated Annealing Algorithm Type, Data points

Where in Data points we collect: time in seconds, cost, iterations and id.

### 03.03.26: Restructuring the Temperature

I looked at the Morpheus implementation and what it refers to `acceptance_rate`, basically for each temperature stage it iterates over a fixed amount of elements per temperature, which is defined to be 100. Out of those 100 elements the acceptance rate is the number of moves that have been accepted. I swapped the scheduling for the morpher cooling strategy and I added a cooling that slows down as time increases and that has an arbitrary max cap of temperature.
Then I moved on changing all strategies to classes and started laying down the architecture and structure of the benchmarking suite.

### 09.03.26: Implementing strategies and Monomap script

I advanced by adding extra functionality to the monomap script, I added a flag `-a` that runs in automation all compiled problems that finds inside `./benchmarks/extra/[**]/[**]/kernel1_edges`, it computes the II if for the given problem does not find the II in the map that holds the benchmarks' II, except for cfd in size 2x2 that is skipped as it blocks the script calculating the II.
Once introduced I had to reduce the complexity of the script, by extrapolating in a `run_benchmark` all different ways to handle a call t the monomap.py script. Moreover I added a jump `-j algoname-X` flag to the script that skips benchmarks until it finds the algoname and size problem.
I implemented all lists strategies beside the loop length heuristic, so: Worst positioned nodes, Worst Positioned nodes and most dependencies, an similar version ordered before by positioning cost and after by dependencies (both did have a higher cost associated with in dependencies and less for out dependencies) and by most clock dependencies (nodes that lies within schedule times that have more nodes than other come first within the list). Then those lists are drawn from with a poisson distribution in three different lambda configurations: fixed, proportional to the size of the list and proportional to the temperature, the higher the temperature is the higher the lambda will be as we decrease the temperature the draw will get greedier.
After having benchmarked those strategies for the first 4/5 problems we found that they do not give any advantage compared to the random strategy, thus for now we move them to the archive.

### 16.03.26: Implementing strategies and Monomap script

finish log
add extra data to benchmarks
adjust slower warmup temperature
start benchmarks

implemente new temperature
benchmark

## Websites used

Collection of online resources used at what time, when and for what.

### Setup

[bits/stdc++](https://stackoverflow.com/questions/28994148/how-can-i-include-bits-stdc-in-xcode?rq=4) 06.02.2026
[r--sysroot $(xcrun -show-sdk-path)d](https://code-examples.net/en/q/4c0b1af/the-macos-sdk-path-why-clang-needs-isysroot-and-gcc-doesn-t) 06.02.2026
[OSTYPE](https://stackoverflow.com/questions/394230/how-to-detect-the-os-from-a-bash-script#394235) 06.02.2026

### Understand the code

[nwteorkx Grpah type](https://networkx.org/documentation/stable/_modules/networkx/classes/digraph.html#DiGraph) 07.02.2026 at 16:29
[python enums](https://docs.python.org/3.10/whatsnew/3.10.html#pep-634-structural-pattern-matching) 08.02.2026 at 12:29
[VF2 Algorithm](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html) 08.02.2026 at 13:06
[GraphMatcher.subgraph_mono](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isomorphism.GraphMatcher.subgraph_monomorphisms_iter.html#networkx.algorithms.isomorphism.GraphMatcher.subgraph_monomorphisms_iter) 08.02.2026 at 13:06
[categorical_node_match](https://networkx.org/documentation/stable/_modules/networkx/algorithms/isomorphism/matchhelpers.html#categorical_node_match) 08.02.2026 at 13:06

### Simulated Annealing Algorithm

[SimulatedAnnealingMapper](https://deepwiki.com/ecolab-nus/Morpher_CGRA_Mapper/4.3-simulatedannealingmapper) source not used (yet)
[SimulatedAnnealingMapper Source](https://github.com/ecolab-nus/Morpher_CGRA_Mapper/blob/b2ec107e/src/mapper/SimulatedAnnealingMapper.cpp#L1-L25) source not used (yet)

[random choice](https://docs.python.org/3/library/random.html#random.choice) 10.02.2026 at 14:30
[temperature update routine](https://github.com/ecolab-nus/Morpher_CGRA_Mapper/blob/stable/include/morpher/mapper/SimulatedAnnealingMapper.h#L172) 12.02.2026 at 10:50
[acceptance_rate of morpheus](https://deepwiki.com/search/devin-look-at-the-simulated-an_dd173aac-0ddd-40c1-96cd-d53915e91121) 03.03.26 at 14:05

#### Temperature in SA

[simple moving average](https://en.wikipedia.org/wiki/Moving_average) 19.02.29 at 15:28
[less computation sma](https://stackoverflow.com/questions/12636613/how-to-calculate-moving-average-without-keeping-the-count-and-data-total) todo

#### Priorities Strategies

[poisson random generator](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.poisson.html#numpy.random.Generator.poisson) 22.02.26 at 16:22 
[numpy.clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip) 22.02.26 at 16:31
[sort a dictionary by value](https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value#613218) 22.02.26 at 19:34

### Benchmark

[python singleton](https://www.geeksforgeeks.org/python/singleton-pattern-in-python-a-complete-guide/) 23.02.26 at 14:34
[python file and path API](https://realpython.com/python-pathlib/) 23.02.26 at 16:07
[python Path to string](https://stackoverflow.com/questions/51847408/getting-string-representation-of-pathlib-object#61639901) 26.02.26 at 16:21
[two scales plot](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html) 26.02.26 at 21:53
[link to files](https://macmost.com/linking-to-files-from-numbers-spreadsheets.html) 27.02.26 at 9:21
[bash strings comparisons](https://www.dotlinux.net/blog/bash-script-string-comparison-examples/) 27.02.26 at 10:16
[how to get directory name in bash](https://stackoverflow.com/questions/3294072/get-last-dirname-filename-in-a-file-path-argument-in-bash) 27.02.26 at 10:58
[how to size graph scales, Text kwargs](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text) 27.02.26
[how to enforce override like methods in python](https://stackoverflow.com/questions/44576167/force-child-class-to-override-parents-methods#44576235) 27.02.26 at 20:26
[how to not specify constructor inputs of parent class python](https://stackoverflow.com/questions/73997582/should-i-repeat-parent-class-init-arguments-in-the-child-classs-init-o#answer-74076953) 06.03.26 at 15:10
[dataclasses python](https://docs.python.org/3/library/dataclasses.html) 06.03.26 at 15:12

### Extra Ideas

[networkx + matplotlib](https://www.geeksforgeeks.org/python/python-visualize-graphs-generated-in-networkx-using-matplotlib/) source not used (yet)
[networkx + matplotlib - 2](https://medium.com/@ruchikahshukla/network-visualization-in-python-using-networkx-8d3dd8657f68) source not used (yet)

[draw DiGraph using matplotlib](https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python#20133763) 09.02.2026 at 13:40
[save figure](https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it) 09.02.2026 at 13:56
[figure size](https://www.geeksforgeeks.org/python/how-to-change-the-size-of-figures-drawn-with-matplotlib/) 09.02.2026 at 13:59

[node nodes](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html#networkx.drawing.nx_pylab.draw_networkx_nodes) 09.02.2026 at 15:55
[node labels](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_labels.html#networkx.drawing.nx_pylab.draw_networkx_labels) 09.02.2026 at 15:55
[font size](https://stackoverflow.com/questions/70673089/python-networkx-pyvis-how-do-you-change-the-node-label-font-size) 12.02.2026 at 12:20

## Questions

What is in your opinion the objective? Make a Simulated Annealing Algorithm and see how it behaves comapred to monomorphism? Perform better than monomorphism? Achieve solutions where monomorphism can't? Or maybe add functionality to the search that a monomorphic search wouldn't be able to consider and use?

> Yes Simulated Annealing should perform better than monomorphism in speed only for larger CGRAs and find solutions to problems that monomorphics can't. While being able to make searches for problems that are imply a more constrained solution that would render a monomorphism inavlid. For example, for the previous EPFL board, where you could only pass the operation result where there has been no operation in between them, for example if op = 1 passes the result to op = 2 scheduled at t = 0 and t = 10, respectively there can't be any operation on the same op = 1 PE until the op = 2 is scjeduled. See [#Roadmap]

## Commands

```sh
# to produce a pdf from the LLVM IR DFG output
dot -Tpdf acc_loop_graph.dot -o dfg.pdfsourc

# -i shoube be the mII in the paper for the benchmark -d defaults already to 5
python3 mapper/monomap.py -path ./benchmarks/extra/particlefilter/kernel1_edges -x 10 -y 10 -d 5 -i 9 -s 1 > cgra-mono-code-particlefilter

# listen to a file writes
tail -n 500 -f cgra-mono-code-particlefilter
```

## Roadmap

### Basic Algorithm Structure

[x] Implement the General Simulated Annealing routine
- [x] Implement random solution generator
- [x] Implement cost function
- [x] Implement temperature and Simulated Annealing Loop
- [x] Implement Neighbour Solution generator function

### Benchmarking

[x] Implement a benchmarking script
- [x] You select the desired algorithm and size, and the script runs it x times (10 at least)
- [x] After each run the script, or the code saves results in a file CSV file
- [x] After each run the code also saves temperature and mappings plot of that run
- [x] If not present the algorithm saves the dfg
- [x] On each meeting we conclude what we see from results on a markdown log: Prepare section
- [x] Move all algorithms to the iteration loop and make them comply with benchmarking: good way to abstract it? Classes
- [x] Improve the code abstraction
- [x] Benchmark already existing algorithms: random nodes and poisson fixed on all extras and all sizes: 2,5,10,20
- [x] Add automted benchmarking? If we can infer II correctly within the code or in bash from the compiled files we can automate benchmarking on multiple problems -> dictionary to predefined collected algorithm name -> II, if not present exit
- [x] Add a way to make it possible to see the current algorithm run, at least the cost / temperature and iterations as to have an idea on at which point it is
- [x] Benchmark poisson temperature

[] Add more data and insights do the benchmark
- [] We add the number of nodes that are positioned correctly and the number of nodes that are not, along a lists of the type: \[source-destination, ..\] for incorrect positioned nodes

- [] We add each node and relative the cost of the solution that they have for the current strategy, i.e. the distance that they have between them, an array of the type: \[source-destination: cost\]

### Algorithm Strategies Implementation

[x] Picking from priority list
- [x] Priority list on worst positioning
- [x] Priority list on most dependencies
- [x] Priority list on most elements in a schedule
- [] Priority list on longest loop

[] Partial solutions
- [] Build partial solutions by following longest loop
- [] Build partial solutions by following the most crouded clock times

[] Random types
- [x] Completely random on available positions
- [] Completely random on available positions and not (swapping)
- [] Random and tabu list

#### Neigbour Function

- [x] Implement strategies of multiple priority list types

   > Heuristics to evaluate, that help choose which one should be moved first to find a good positioning before other operations

- [x] Priority to schedules times that have more nodes than others

- [x] Priority to bad positoned operations

- [x] Priority to bad positoned operations with random number generation from a Poisson Random variable that follows the heuristic: The more chance (to be moved/changed) to the elements that the heuristic defines "to be changed" and less chance to elements that the heuristic defines as "good enough", so that you still give some room to good placed operations that might be needed to find the minimum but you provide more solutions that move elements that are placed wrongly

- [x] Priority to more dependencies operations

- [] Make a SA algorithm that uses the loop heuristic: priority to loop dependencies rather than direct dependencies

   > As the dependency loop is larger the more difficult it will be to make it work within PEs position, thus the more weight it should have: the faster it should be positioned

- [] Construct the solution operationally incrementally: add first 50% of operations (maybe the one with more dependencies) find good placement for them, add the remaining 50% and find a solution with the remaining (fixing or not fixing the first half? Give a small chance to be moved?)

   > Incremental solution construction

#### Temperature

- [x] Modify Temperature to be at least aware of the cost of the current solution: Cristian suggested to see if when we are going to be freezed we are very close to the zero cost solution. If yes, add extra temperature to complete the job

   > Right now we are not sure that the computation is going to terminate even if we know that a solution exist as for more intensive problems, such as the particlefilter, we might need more time than given. Morpheus uses the acceptance probability to decide how to decrease the temperature, we could look at first, simply to keep temperature a little bit hot to keep it running until we make considerable improvements between iterations.

- [x] Simplify SMA computation: Just an average of the last x results

- [x] Modify the temperature to be detached from the iteration cycle, make the algorithm iterate a fixed amount of time and make it 

- [] Make the temperature follow the cost like a MA: make it decrease a little if the last iterations have been good iterations (we improved) or decrease a little if the opposite. Maybe may be good to make temperature start at the cost and then diminish as we approach 0 and freeze at 0 (integers), since it is not a problem to iterate for long, not as a float 100-0.0001.

- [] Add reset temperature completely to the beginning when stuck to random and start benchmarking

- [] Implement a Hybrid strategy: As the temperature gets colder the startegy has more chanche, or switches copletely, to a more heuristic startegy than before, so that it points more to the refinment of the solution as the temperature decreases.

- [x] Apply the temperature in three different ways to the algorithm:
   1. The temperature is used only to decide whether we pick or not a worst solution (hill climb)
   2. The temperature is used also to change the distribution of Poisson's lambda, the greather the temperature the greater the lambda, the less heuristic the search will be
   3. The temperature is used to change the distribution peak of a Gaussian distribution (variance and probability), the more high it is the less heuristic the search will be

#### Cost function

- [] Start by a solutions that already makes sense: construct the solution so that it makes sense to be similar, allow for PEs to be scheduled multiple times on the same PE, as this allows for a low cost solution that still might not be optimal, then when computing the solution we make overlapping nodes "fall to nodes in the vicinity" (then it can be implemented also without changing the cost function) or your solutions are all allowed to take overlapping PEs as a constructed solution

> This strategy needs the cost function to be enhanced as to handle also overlapping nodes cost

- [] After implementing the overlapping might be a variation to allow of a neighbour solution to be picked with overlaps if the temperature is high and do not allow if it is low, like the probability to pick a worse or better solution

### Reflection Over Strategies and Refinement - Iterative Process of "Algorithm Strategies Implementation"

- [] Understand how SA strategies work in depth related to the current problem

   > What are distinguishing factors of bad and good neighbour functions? What are distinguishing factors of a fast or slow neighbour function? Why is one strategy faster then the other? What are key traits that give an advantage to one strategy compared to other?

- [] Comprehensive SA Analysis

   > Do 100 benchmarks on most key problems, colect the time, plot them and compare their median, note that the strategy can't reach the speed of monomorphism and state of the art, but the objective is to reach a robust strategy that might take time, but that both, finds a solution and is consistent on the running time on all benchmarks

- [] List and explain in detail how each strategy works towards a solution

- [] Analyze the behaviour of temperature and cost based on the strategy

- [] Think of both: best traits and worst traits for each strategies

- [] Pick and refine the best strategy that will be the product of the research

### Uninformed VS Informed SA

- [] Uninformed VS Informed SA
   
   > Write an unformed SA algorithm, i.e. a simualted annealing algorithm that searches for both time and space solution at the same time (classical problem), then use your informed simulated annealing, the one that searches the spatial solution after knowing the time solution and that it exists (decoupled CGRA mapping) and comapre the two algorithms in comuting solutions, should appear that the second one is far more better and capable.

### Different Architecture Problem

- [] Modify the Simulated Annealing algorithm to take into account of a different architecture. Christian worked with a CGRA chip architecture that does not allow for the scheduling of an operation on a PE that has already run an operation and such operation output has not been delivered yet. Put more simply: The current MRRG for each PE has an edge to the same PE of all clock times, in that architecture it is not true, if you schedule an operation on PE 0 at clock time 0 you can't schedule an operation on PE 0 at clock time 1 that does not take as input the ouptut of operation at PE 0 on clock 0, it will "consume" the output and it can't be delivered anymore to the right operation.

   > It will introduce a new cost function calculation and a new check function to verify that the space solution is valid

## Meetings

Notes on the discussion of each meeting with Cristian

### 23.02.26

la temperatura veniva usata per scegliere quali nodi vengono spostati: più è alta più è possibile avere mosse non ottimali

troppo greedy: 

posizionare path uno alla volta e cercare una soluzione poi aggiungere path e così via

più tempo spendi nell'algoritmo meno è casuale

se le ultime 100 soluzioni hanno la stessa media alzo la temperatura per cercare di scappare dal minimo

benchmark tutti i problemi 2, 5, 10, 20

gaussiana che con temperatura alta inizia a destra (nodi meno importanti: meno costo meno dipendenze) e piano piano si posta verso l'inizio con la temperatur (tipo poisson con lambda = 0)

mettere x iterazioni e staccare la temperatura in modo che sia colloegata a come scegli le soluzioni e la probabilità di accettare soluzioni peggiori: grafico diventa cost / iterazioni, più pulito

partire con una soluzione che abbia senso: che sia posizionata con un heuristico

lista di operazioni non pronte (non assegnate) e le assegni mano mano una alla volta

ordine topologico di selezione dei nodi

script bash che colleziona i dati delle varie run
file di overview: in cui elenco tutti gli esperimenti fatti e come gli abbiamo fatti, con data e data del meeting

### 02.03.26

scegliere il primo nodo nella lista di peggiori invece di poisson

- lista: distanze con le dipendenze, edge in output (piu ce ne sono più sono importanti), loop cicles maggiori

spikes di temperatura sono troppo elevate: la temperatura va gestita più "vicina" al costo delle soluzioni, la discesa deve essere meno aggressiva

movimento più docile della temperatura e senza spikes, deve crescere "linearmente" in base al costo

usare moving averages per gestire la temperatura: in diminizione diminuiamo la temperature, mentre se rimane stabile per molto tempo sma201 < sma5 aumenti la temperatura regolarmente per stimlare hill climb

partire da soluzioni non a caso: soluziondi già calcolate in maniera heuristica, a random svantaggia con cgra più grandi: prendere operazioni una alla volta ordina il graphico in maniera topologica, parti da uno e man mano scegli nodi aggiuntivi che sono collegati e li posizioni già giusti: anche se sono in due PE uguali, dopo usi la funzione costo con due costi: overlapping PE allo stesso tempo, e distanza, poi cerchi soluzioni avendo anche quel costi di sovrapposizionamento finchè non trovi una soluzione

legare overlapping alla temperatura? più è alta più è possibile selezionare soluzioni con overlapping

overlapping non sembra avere senso nella cerca della soluzione: sappiamo che cn la schedule non ci saranno overlappping: comunque da provare

poisson independente dal numero di elementi nella lista: 5 massimo

aggiungere costo della soluzione mentre calcola così da vedere a che punto siamo

next week:

experimenti benchmark tutte le dimensioni a algorithmi
aggiustare temperatura che segue il costo come detto (un'occhiata all'implementazione di morpheus)

### 09.03.26

Idee per la nuova temperatura:

- Temperatura ugualle alla MA +- epsilon random dipendente dal costo: Costo basso più epsilon è basso 
- Fai seguire la temperatura con la media mobile (no freezing temperature), e nel riscaldamento al massimo più di 50% di differenza dalla MA
- MA +=5% e vedi come si comporta

Prima: aggiungere extra dati ai risultati, poi fare benchmarkings:

1. benchmarking con morpheus (da random) e no riscaldamento + random: almeno uno per algo e dimensione

2. benchmarking con morpheus (da random) e riscaldamento attivato con MA (vecchio: Temperatura usa MA per decidere quando scaldare, modifica: riscalda molto più lentamente) + random: almeno uno per algo e dimensione

3. benchmarking con temperatura MA (nuovo) + random: almeno uno per algo e dimensione, provare prima run indipendenti per trovare il migliore, poi benchmarks

benchmarking con morpheus (da random) e riscaldamento con Altre idee (pensa a come relazionare temperatura con MA/Costo) per temperatura + random: almeno uno per algo e dimensione

Idea:
invece di essere random prendere un nodo e calcolare tutti i costi delle soluzioni: poi prendere il pe che minimizza il costo (tanto greedy)... TODO dopo

NON implementare, ma possiamo pensarci e definire come si potrebbe implementare:
Fai una lista di soluzioni: se la temperatura è in discesa scegli quelli che minimizzano il costo, in salita scegli quelli che non minimizzano (a livello probabilistico) -> derivata della temperatura (che è definita come moving average, per capire se diminuisce o aumenta il costo)... TODO dopo

per la costruzione di una soluzione:
invece di andare a caso guardare il neighbour con la funzione costo e scegliere la posizione che minimizza il costo... TODO dopo

mettiamo da parte liste

dati risultati
aggiungo numero di nodi che sono posizionati giusi e numero posizionati correttamente
aggiungere le distanze che compongono il costo

### 17.03.26

TODO
