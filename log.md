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
From [morpher](https://github.com/ecolab-nus/Morpher_CGRA_Mapper/blob/stable/include/morpher/mapper/SimulatedAnnealingMapper.h#L172) they update the temperature based on acceptance rate, let's see how it behaves in our system.
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

I looked at the Morpher implementation and what it refers to `acceptance_rate`, basically for each temperature stage it iterates over a fixed amount of elements per temperature, which is defined to be 100. Out of those 100 elements the acceptance rate is the number of moves that have been accepted. I swapped the scheduling for the morpher cooling strategy and I added a cooling that slows down as time increases and that has an arbitrary max cap of temperature.

Then I moved on changing all strategies to classes and started laying down the architecture and structure of the benchmarking suite.

### 09.03.26: Implementing strategies and Monomap script

I advanced by adding extra functionality to the monomap script, I added a flag `-a` that runs in automation all compiled problems that finds inside `./benchmarks/extra/[**]/[**]/kernel1_edges`, it computes the II if for the given problem does not find the II in the map that holds the benchmarks' II, except for cfd in size 2x2 that is skipped as it blocks the script calculating the II.
Once introduced I had to reduce the complexity of the script, by extrapolating in a `run_benchmark` all different ways to handle a call t the monomap.py script. Moreover I added a jump `-j algoname-X` flag to the script that skips benchmarks until it finds the algoname and size problem.
I implemented all lists strategies beside the loop length heuristic, so: Worst positioned nodes, Worst Positioned nodes and most dependencies, an similar version ordered before by positioning cost and after by dependencies (both did have a higher cost associated with in dependencies and less for out dependencies) and by most clock dependencies (nodes that lies within schedule times that have more nodes than other come first within the list). Then those lists are drawn from with a poisson distribution in three different lambda configurations: fixed, proportional to the size of the list and proportional to the temperature, the higher the temperature is the higher the lambda will be as we decrease the temperature the draw will get greedier.
After having benchmarked those strategies for the first 4/5 problems we found that they do not give any advantage compared to the random strategy, thus for now we move them to the archive.

### 17.03.26: Implementing strategies and Monomap script

We started by adding extra information to benchmarks, we added pair of nodes that were not correctly positioned so that we can spot which nodes are not in position. We also added the cost of those out of position nodes and the total quantity of correctly positioned nodes and not correctly positioned nodes.

We added different kind of temperature strategies as we wanted to test which one seems the most promising. We introduced these temperature algorithms:

- Cooling: Which reduces the temperature after we have seen enough elements by a constant factor $\alpha < 1$, it was set to $\alpha = 0.9$, it also defines a fixed number of items to be visited each time equal to the number of nodes times 100.

- Temperature SMA: We move temperature proportional to the moving average of the cost plus or minus a delta. We start by aligning the temperature to the SMA of the cost minus the delta as to make the temperature decrease and solutions to improve. Once we get stuck: We have two other SMA a 200 items SMA and 5 items SMA and by stuck we mean that the 200 SMA is smaller the 5 SMA plus a small epsilon, we increase the temperature SMA to be proportional to the SMA cost of the cost plus delta plus a small epsilon that increases as we complete warming iteration cycles. As we reached an arbitrary fixed amount of warming iterations we stop warming up and we slowly reach SMA cost minus delta for the temperature.

- Morpher Reset SMA: We keep two costs SMA with windows of 5 and 200 items. We then proceed cooling the system using the acceptance rate the same way Morpher does, and when the 200 SMA is smaller than SMA 5 plus a small epsilon we reset the temperature to the starting value (arbitrary and of 100). If the temperature reaches a cooling temperature (arbitrary) we stop cooling.

- Morpher: We decrease temperature following the Morpher schedule and we stop at freezing temperature. No reheating.

- Morpher Warmup SMA: We keep two SMA of 200 and 5 items window. We then reduce the temperature following the morpher schedule, and if the are stuck (200 SMA less than 5 SMA plus epsilon) we reheat, reheating follows a slow at the beginning and faster at the end schedule, the schedule is arbitrary and has no correlation with the system structure and state other than using the number of warmup iterations.

We tested those temperature rutines with two, very similar, algorithms:

- Random: We select a random node to be moved, then within its schedule we select an available PE to move the node to. An available PE is a PE that the selected node can be moved to without overlapping with other operations (its current PE excluded)
- Random with swap: We select a random node to be moved, then within its schedule we select a random PE, only its current PE excluded from the selection. If the PE is free we move the node, otherwise we swap it with the current assigned node, ensuring no overlapping of operations.

Added legend to graphs

After discussion, we made the decision to discard the SMA temperature strategy and to move on with SMA reheating, cooling with added spike and morpher with spike as we cleary see that reheating is needed to obtain much better results. We saw no substantial difference between Random and Random with swap strategies.

Another idea I thought about is that we can't understand how good a valley is, or whether we left such valley (probably) but we might be able to udnerstand earlier if we are in a valley that is not that good, so that if we fall again into it we can steer away the search without going down the valley, saving us some extra work: would it be possible to understand a bad configuration with the architecture and nodes placements? Such a structure may be defined as one that needs multiple moves to make it possible to reach a viable solution. We will implement a version to see if it improves our current strategies.

We should improve also the plot of the cost/temperature, iterations.

### 23.03.26 Adjusting Time Problem

Added additonal string to the output file using the flag `-l` to be able to distinguish different running processes of the same algorithm.

Added seeds for both the creation of a starting solution and the randomness of the search algorithm, the creation seed is set by default to `12345` and the running algorithm is set to a random seed (no specified seed), both are recorded in the benchmarks information. We do so because we want to fix the start configuration, as we can exlude the fact that a benchmark run has been luckier than others, thus making a more even field for benchmarks.

Adjusted format for benchmarks incorrect positioned nodes' costs and better formatting for execution times.

Made baseline algorithm for temperature and items handling so that benchmarks can be analyzed to find the time problem. Added routine time information to benchmarks.

Improved temperature and cost plots: We make 4 plots: A plot of all iterations, a plot with iterations averages into windows such that `max_elements` are the most number of elements retunred, a plot that divides iterations into three subsets and plots them one under the other and a last plot that takes all the temperature iterations of the cost, normalize them into [100, 1] cost values and plots them on the same axis one over the other, showing the evolutio of the graph.
This step aims at providing a better graph visualization of the temperature and cost, goal is to then reduce graphs to the most helpful set. Changed benchmarking code to save all figures in a folder for each run.

### 30.03.26

I started by completing the benchmarks for:

- morpher + spike and reset of MA to initial value: 100 items slow MA
- cooling + spike: 100 items slow MA and ITEMS_PER_TEMPERATURE = number of dfg nodes * 10

---

Note that I still find some discrepancies of time and iterations within the same algorithm, for example:

| algorithm | dfg_nodes | size_x | size_y | sa_algorithm_type            | time_seconds | cost | start_configuration_cost | iterations | seed_start_configuration | seed_algorithm_run |
|-----------|-----------|--------|--------|------------------------------|--------------|------|--------------------------|------------|------------|------------|
| aes       | 23        | 10     | 10     | COOLING-RESET_RANDOM-NODE    | 453.63       | 0    | 817                      | 347        | 12345 | 59226991 |
| aes       | 23        | 10     | 10     | COOLING-RESET_RANDOM-NODE    | 329.66       | 0    | 817                      | 111        | 12345 | 40226105 |

Yet I run both settings using time ..script.. and the time is precise, also their average execution values are correct, in which they take respectively, 1.31 (1.31 s * 347 iterations = 454.57 tot s) and 2.97 (2.97s s * 111 iterations = 329.67 tot s) seconds to look at all items (time of one iteration), which comprehends: solution generation, solution cost computation and acceptance check all for x items. Yet they see the same number of items and of the same problem and configuration, which begs the question: where does the difference of time come from?
Both have a average_cost_space_sol_time_item of 0.01, and average_neighbor_sol_time_item of 0.00, which they compute the cost of the solution and construct a neighbor solution, which given 230 items gives roughly 2.3s of each iteration, which makes the faster running algorithm on average faster of about 1 second on the whole iteration and the slower one slower of about 0.67 seconds on the whole iteration (for both considering only the generation, cost computation and scceptance check time cost). There is no data there that can help an assumption on this matter, first I want to cinfirm again that closing the laptop and opening it after does not produces imbalances in times. Even is these two cases have not experienced it, they have been collected at: 21-41-40 and 21-49-16 (the time the script is called) and the subsequent one is the aes 20x20 started at 21-54-48, the delta bewteen the first 10x10 and the second one is 456s and the 10x10 with 20x20 is 332s, which coincides with the benchmarked times.
I tried stopping the machine with time (close then reopen) and i get:

./monolang -p ./benchmarks/extra/aes/kernel1_edges -x 10 -y 10 -s 1 -l _aes59226991  579.82s user 1.53s system 3% cpu 4:49:56.05 total

./monolang -p ./benchmarks/extra/aes/kernel1_edges -x 10 -y 10 -s 1 -l _aes40226105  456.95s user 1.36s system 2% cpu 4:47:52.83 total

for our problem with same seeds respectively, which definitely adds noise to benchmarks, not that much as using `.time()` but considerably enough, for our purposes I think it is fine when not collecting data to show in the report. Both times in th script and time computed by `time` are equal also in this situation.
It comes down to the solution cost: `cost_space_solution`, as we can see that it is the routine that makes for more time. If we nail down what within such execution can go wrong we can show that we are looping over edges of the dfg, which are probably saved as a dictionary within the DiGraph class (`self.dfg.edges`) then at the same time we use a few dictionaries and `isConnected` which is only a few O(1) operations. What may make the difference is actually how we compute the distance between each node, as we use `shortest_path_length` of networkx we may see discording computatio times if they are positioned, so instead of calling it each time, as the CGRA is fixed overtime we can compute it before we start execution and then use the dictionary to retrieve the cost of each node composition. It turns out it was the discrepancy reason and made for a much much slower algorithm run, now they both return an average time of:

./monolang -p ./benchmarks/extra/aes/kernel1_edges -x 10 -y 10 -s 1 -l _as59226991 13.23s user 0.32s system 98% cpu 13.751 total

./monolang -p ./benchmarks/extra/aes/kernel1_edges -x 10 -y 10 -s 1 -l _as40226105 11.94s user 0.23s system 99% cpu 12.250 total

which looks dominated by the preparation time, where we compute all CGRA distances since we got a simulated annealing time of 1.86 and 0.60 and a total iterations of 347 and 111 respectively. Which is way more proportional as 0.6 * 3 = 1.8 for 333 iterations.

Still I am sure we can compute it on the fly using only arithmetics instead of precomputing the entire table for the whole CGRA, because there are enough fixed informations, the same way we can compute whether two pes are connected. Solution is: if we have both the row and the column indexes of the two pes, we can get the distance within board in terms of column and rows edges by taking those indexing and computing the distance. Then to handle wrap around we know that the path for each axis from one point to the other is a circle because we are connected wrapping around edges, and the size of the column / row of the CGRA is the number of total edges we need to traverse to go from one node to the same node wrapping around, so if we subtract the distance row wise or clumn wise from nodes within the edges of the board we will get the distance of the nodes if we were to wrap around and if we take the minimum distance between wrapping around and within board from each column and row distances we will get the minimum distance from one pe to the other.

+ + it is good if I add a test suite to check the correctness. Using [unittest](https://docs.python.org/3/library/unittest.html).

---

I added `__init__.py` so that python will treat directories as packages, because if not unittest doesn't move to all directories during discovery. I'll leave `/archive` away as they are just folders to keep old unused code.

---

Another very odd situation is the new routine concept:

We select the next node from a list already prepared with all nodes, and the next pe the selected node will move is drawn from a list of already prepared pe indicies. The problem is that given this setting the node selected and the pe selected will make it swap with itself, which should be a problem since it is not that statistically significant also because it is unprobable that it will continue to so. Long story short, given this implementation the algorithm times out and is unable to solve even a 5x5 aes problem, which does not imposes a lot of difficulty for random node. Moreover, if we add a while loop to keep selecting a new pe until we select one that is different from the current node to move pe (if we select it in the first place) we see that the algorithm is able to run well: the statistically insignificance seems to actually be significant. I plotted also hisotgrams of before and after of the random selection of nodes and it follows a iid distribution, all nodes and pes are selected randomly (before we add the `while` routine, which means no implementation errors that might have biased results). I also added per node draw of `new_pe` histograms and they follow.
I thought that maybe it might be to the descent speed of the temperature: given a reduction of `0.9` the descent speed is so fast that the probability to pick up an item that is my exact same can influence the search and make it lose key neighbor solutions at each temperature level. I tried reducing the temperature pace but it is still unable to come to a viable solution.

Well there was an error:

```py
curr_node_pe: dict[int, int] = copy.deepcopy(self.node_pe)
curr_pe_nodes: dict[int, list[int]] = copy.deepcopy(self.pe_nodes)
```

The assignment `curr_node_pe: dict[int, int] = self.node_pe` does not copy the object, it assigns the reference to the object, while `.copy` copies the first layer of the object, for a list of objects it copies the list of references but not the copy itself, while `copy.deepcopy` makes a copy of all layers. As a reminder: "All objects in Python. Assignment never copies.".

Immutable built in types are:

- `int`
- `float`
- `bool`
- `str`
- `tuple`
- `NoneType`
- `bytes`
- `complex`

I did extra benchmarks with improved copying strategies but at the same time we can't say that the new routine is faster, compared to the deep copy it definitely is and that one has to be discarded completely but compared with the morpher routine which loops over the ds and copies them over for same starting seed and different algorithm runtime we can see that sometimes it does beat morpher but sometimes it does not.
To make a fair benchmark I should operate them on both the same starting solution and the same algorithm seed: I did so using time without adding results to benchmarks and I got:

./monolang -p ./benchmarks/extra/backprop/kernel1_edges -x 20 -y 20 gave 45.49s user 0.80s system 99% cpu 46.455 total

./monolang -p ./benchmarks/extra/backprop/kernel1_edges -x 20 -y 20 gave 26.20s user 0.72s system 99% cpu 27.187 total

for new routine and old routine respectively, and also for the random algorithm run seed which the new routine performed better we got:

./monolang -p ./benchmarks/extra/backprop/kernel1_edges -x 20 -y 20 gave 17.58s user 0.42s system 98% cpu 18.336 total

./monolang -p ./benchmarks/extra/backprop/kernel1_edges -x 20 -y 20 gave 6.36s user 0.25s system 97% cpu 6.797 total

for new routine and old routine respectively, which clearly marks the old routine being faster than the new one: iterating over each data structure and changing the solution while doing so is faster than coping entire data structures and applying the change after, but of course the latter has simpler code structure.

---

I then went to manipulate the equation of the temperature to be set in such a way to produce an exact percentage of probability: From $P = e^{-\frac{\Delta c}{T}} = 0.3$, where $\Delta c = \text{solution cost} - \text{best solution cost}$, I inverted in such a way that $T$ would be represented as positive value since it is the case that we have two solutions: $-T$ and $T$ that lead to the same result. I arrived to:

$$T = \frac{\Delta c}{\ln{\frac{1}{0.3}}} = \frac{\Delta c}{\ln{\frac{1}{P}}}$$

Still it remains that $T$ is a function of two variables: $\Delta c$ and $P$, which $P$ can be set easily by us as the probability that we want to get but $\Delta c$ is problem specific and ranges widely: the larger it is for the desired $P$ the larget $T$ will be, because we say that we want a probability of $0.3$ given much worse solution, while a $0.3$ percent of probability of being accepted for a solution that is not much worse $T$ can be much smaller.

Still it can be used at our advantage as it follows the first benchmark that we wanted: if we compute the average cost of the current items visited, if we are in a plateau that is bad, the $T$ computed using the average for $\Delta c$ will be much greater, leading us to to move more far away, while opposed for a good solutions plateau we will increase it moderately.

See that a way to fix it could be using the starting solution as baseline solution cost, and use the current best solution to compute the temperature that we should set to get the exact $P$ we want.
Or to not be biased by initial solution we could compute the maximum possible cost of such problem with that architecturte, but it is more difficult to implement.
Another way could be to collect information about probability and its relative temperature during descent so that it will be balanced with the solution cost that the problem sees at that temperature.
We could also skip the temperature completely and jump straight to managing the acceptance probability and make it descent at fixed intervals instead of moving through the temperature factor, independently from delta cost, or adding also a dependence there.

I implemented them and tested on several single executions of `hotspot3D` with a 10x10 architecture which gives a balanced problem, which may be in the vicinity of cfd as difficulty but, which usually, does not take up to timeout to find a solution.

Both fixed methods using the highest cost solution of the architecture and the start solution as $\Delta c$ gives too much temperature, so much so that `0.3` sets temperature to a one higher than the one we started with.

The dynamic temperature computed is such a way that $\Delta c$ is the difference with the average visited solutions minus the current solution cost provides a more balanced reheating: With a seeked percentage of 0.3 the temperature is set to give around 0.5 percent, which of course is not the one we hped for but does not make the temperature fly as previous algorithms. While using directly the current solution cost yields a probability that is aligned with what we were looking for. I still want to try and implement the learned temperature strategy to see how it behaves

I started benchmarkings with different reheating probabilities. A few throught about these benchmarks: The run does not produce always the same result, this might be because, even is the random number production is the same for all runs, temperautre is not: at some levels there might have been accepted some solutions that for certain run they have not, changing the course of the run even if randomness is exactly the same. Think of a solution of acceptance probability of 0.27 and the two run that reheat to 0.2 and 0.3, one will not accept it the other not. I did first hotspot3D 10x10 before moving to 20x20 so that I have a middleground of times to have an idea how temperature behaves on a more conservative benchmark.

The strategy I choose to use as baseline for benchmarkings is the one where we use the current best solution cost directly as $\Delta c$ it introduces a kind of Dynamic reheating: as the plateau valley is less coslty the percentage that is returned for such solution to be accepted is smaller than one which is greater, because logically it is returning the temperature at which such "delta" is being accepted with the given selected probability. As soon as I finish the fixed cost I can benchmark the learned probability, which should give a much more precise reheating and we can add to it a dynamic reheating based on the previous plateau: we compute the difference from previous plateau and add in some way more temperature if it is greater or equal than before, and maintain to fixed probability if smaller. This should address the dynamic temperature startegy.

We might have found a random seed that performs well and finds solutions also for the 20x20 with relative ease, I have two seeds that do not and that have been already benchmarked (although not by using this algorithm strategy). It might be a good thing to use also such seed to benchmark temperatures.

Then we can move to the different model number using the fixed probability reheating that seems working best. We will use the reheating to fixed probability with best solution cost with a probability reheating of 0.5 which seems to be the most promising and balanced. I am going to test it on hotspot3D at 20x20, which is th hardest problem before cfd. I have just two doubts:

- I hope the size means the number of nodes and not the number of edges.
- I hope the algorithm run seed for the 9 T benchmarks should not be set to always be the same as nothing is said, if i get time I will do also with fixed algorithm run seed

I need to ask if the number of edges and maybe distinguish the type such as direct dependency or loop dependency might be also good to consider.

After when we understand the best starting T I will make a total benchmark with reheating to porbability and the new T start.

T start temperatures benchmarks show that 

---

So all things done are:

- fixed time inconsistencies made by shortest path
- fixed new routine deep copy problem
- redone all old benchmarks
- changed plot y to probability instead of temperature
- added a record functionality to save the entire run data (costs, temperatures, SMAa)
- Implemented different kind of temperature to exact probability reheating ($\Delta C$, explain thought behind it):
   - Starting solution, removed
   - Worst estimated solution, removed
   - Average solution cost, removed
   - Current best solution cost, kept: best
   - Learned during descent, kept
   - Fixed probability detached from temperature, kept but we can probably remove it, discuss

- Questions: ...should we consider edges and type for start T? ...should we make benchmarks also for items coefficient the same we did for the start temperature?

So all benchmarks made are:

- morpher + spike and reset of MA to initial value: 100 items slow MA
- cooling + spike: 100 items slow MA and ITEMS_PER_TEMPERATURE = number of dfg nodes * 10
- new routine with deep-copy and new routine with improved copy: we want to checkspeed against the old routine
- Benchmarked with different % on a hard problem (hotspot3D, as we can solve it but sometimes we don't) with the most promising temperature system: Current best solution cost, with same algorithm routine, same start and algorithm seeds. Done hotspot3D with random seed and seed we failed for morpher at 20x20, did an additional run for hotspot 10x10 with the random seed. % run from 0.05 to 0.8 at 0.05 % step.
- Benchmarked with different % on a hard problem (hotspot3D, as we can solve it but sometimes we don't) with the most promising temperature system: Current best solution cost, with same algorithm routine, same start and algorithm seeds. Done hotspot3D with random seed and seed we failed for morpher at 20x20, did an additional run for hotspot 10x10 with the random seed
- Benchmarked the set of 9 benchmarks using the cooling and reset probability set to 0.5, with the exact set settings and random algorithm run seed for all T made of number of dfg noes times 1, 10, 100
- Benchmarked the set of 9 benchmarks using the cooling and reset probability set to 0.5, with the exact set settings and random algorithm run seed with a fixed algorithm run seed for all T made of number of dfg noes times 1, 10, 100

### 13.04.26

/Users/eliariva/Documents/Puddi/USI/BachelorProject/MonoMap/benchmarks-results/hotspot3D/20x20/COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4/2026-04-11T13-25-29

I have recorded the data of the run and the result is: 

| iteration | costs | temperatures | probabilities | costs_sma_fast | costs_sma_slow |
|-----------|-------|--------------|---------------|----------------|----------------|
| 0         | 2901  | 57           | 0.18077907358023343    | 6445.0    | 7286.7       |

and indeed it does not seem to start at the right temperature cost.

/Users/eliariva/Documents/Puddi/USI/BachelorProject/MonoMap/benchmarks-results/hotspot3D/20x20/COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4/2026-04-10T21-43-01

for this benchmark we have:

I have recorded the data of the run and the result is: 

| iteration | costs | temperatures | probabilities | costs_sma_fast | costs_sma_slow |
|-----------|-------|--------------|---------------|----------------|----------------|
| 0         | 8331  | 5700         | 0.9757668062431737  | 7531.0    | 7341.0       |


and indeed it does not seem to start at the right temperature cost.

The plausible reason I can think before looking at the cose is that since the SMA record a cost very close to the actual start plus next iteration element, I probably don't save the first iteration cost, I save if after we select the first solution.

Indeed, the issue was that I was nto adding the very first solution that we find, once added the two starting iteration of the two run are, respectively:

| iteration | costs | temperatures | probabilities | costs_sma_fast | costs_sma_slow |
|-----------|-------|--------------|---------------|----------------|----------------|
| 0 | 7331 | 57 | 1 | 7331.0 | 7331.0 |
| 0 | 7331 | 5700 | 1 | 7331.0 | 7331.0 |

Still, the chart does not look like the recorded data. As we put the log scale the problem is that `log(0)` is not defined, thus matplotlib drops the first value without any warnings whatsoever, so it appears as if it is not there, we can solve it by making the x range start from 1 up until total iterations. After fixing it I also removed noise from the probablity by computing the acceptance probability only if the selected items made the solution cost change, this is because it is very improbably that at high temperatures the probability is set to the same value while at low temepratures it is very probable and it will happens to accept a new solution rarely, thus we should be able to mitigate the noise by not adding probability when such case happens, and in case it does at high temperatures we bring forward the last probability, which should be close to the current one and ultimately it won't mess up the chart.

I think it does in fact reduce the noise by a lot improving the readability of charts while maintaining correctness.

---

It is hard to set the temperature to a fixed probability because the problem it is in varies a lot, and the probability to pick up a solution always depends on the delta of the two solutions.
Picture a 5x5 and a 20x20 the same probability that will give a 50% acceptance rate in a neighbor is different between the two sizes, moreover it differs also in the quality of the given solution. To iterate better I re-implement previous algorithm and benchmark them all for probabilities: `0.5`, `0.45`, `0.4`, `0.35` of expected reheating probability to show how they behave when handling the reheating. Strategies implemented and tested are, and they are labeled dependning on how the $\Delta c$ is selected:

- Best solution until now
- Learned during descent rehating (actually, here we do not learn $\Delta c$ but to associate the temperature to the probability)
- Fixed probability
- Last improvement delta (new)

and benchmark them all on hotspot3D 20x20 for something like around 5 benchmarks each.

I implemented the last improvment delta, but at a very similarity of same reheating c with the best solution strategy it does diverge towards less costs temperatures and is unable to reheat up to the probability we have set, in contrast the best solution does reheat more. A difference has to be noted: the last improvment delta plateaus at around a cost of 150 while best solution lower, or when it is higher and would plateau at 180 it will use a c of 180 to reheat.

Best solution until now for a solution cost of `4` applies:

`applying temperature rehating to: 5.7707801635558535`

of temperature when reheating and the probability shoots up as to accept solutions of cost of around 80/50 with maximum of more than 120, still the temperature probability in the chart seems misleading to me.

Similarly, at low temepratures the delta is very low even for which solutions costs:

`applying temperature rehating to: 4.328085122666891`

as it probably accepted a very tiny improvment of 4 or 1. But when reheating such temperature does not make it able to escape the local minimum as it would need a very high temperature to accept worst solution it finds itself blocked as such low c provides very low temperature, and as there are no better solutions in that neighbor it will never be able to escape. Since it timeouts ofter I will do less tests for such strategy.

I spent some more time iterating over the fixed temperature strategy as it did not worked even closely to how it should, indeed there were some problems such as the missing discouraging of bad solution and a cooling that did not reached colder enough temperature for enough time. 

After doing so I feel more confident about the dynamic best solution reheatign with a reheating to 0.45 of probability. I feel like in benchmarks it behaves much more consistently compared to other solutions.

---

I refactored the class strucutre to comprehend a start function that is implementation based, and I extraploed from strategies up to a root to `builder.py`, in which given a class for a start, routine and strategy you can make the composition of a simulated annealing strategy as to make a construction with those different classes easily.

---

From the tests of all benchmarks with a startign temperature coefficient of 1, 10 and 100 I found that usually the 1 coefficient runs are faster than others but by consistency 10 perform bettter (backprop, basicmath, crc32, gsm, stringsearch), in some benchmarks 100 temperature also performed better (gsm, hotspot3D), but I think at this level of closeness we need more data to be sure about it. 

I found also a very strange behaviour for a stringsearch problem, 20x20 where the search seems to be stuck and the temperature to never reheat, with the 100 temperature start coefficient. The seed is: 64051248. I will look into it

---

I am implementing the starting algorithms. While implementing them I got an idea for a new routine that might better for problems which happen to be stuck a lot in local minima. Strategies are:

- Opening: We select the miggle PE in which we position all nodes, then we iterate over each node and select one of then neighbouring positions: top, right, bottom, left, and we move it until we end un in a position that does not overlap with any node. After doing so with all nodes we obtain a configuration that is very close, ensuring a lower cost.
- Greedy: We first define a function `nodes_pes_overlap_count` that iterates over all nodes and counts how many overalps there are for each node, since overlaps will be counted twice we then divide by two, returning effectively how many distinct overalps there are. While computing overlaps we build a list of all nodes indexes that are overlapping. At the beginning we position all nodes in the middle PE, we then compute the overlap count which is at the maximum possible value, and we enter a while loop that terminates when overlaps reach zero. In the while loop we select one node from the overallping list, I then iterate over all pes for that node and I will select the pe out of them all with both the least cost and least overlapping count, then apply it, recompute overalpping costs and list and iterating again. Once the overallping count reaches zero we stop.
- Temperature Start: We start at high temperatures and we position all nodes within a unpositioned list of nodes, we also keep an overlapping count and a list of overlapping nodes at each iteraton. We then take a node from one of the two list at random: either unpositioned or overlapping or if one of those is empty we take the other list, then we select a rnadom node within the list and the positon and check: if the solution and overlapping costs are less than they were previously we will accept right away otherwise we go with the temperaute probability and look to accept based on a probability.
- Temperature Start Inverted: We start by a list of unpositioned nodes, and by maintaining a list of overlapping nodes at each iterations. We then start and select from one of the two lists at random but that decays to the non empty list if one is empty, a random node and a random pe position, then we accept it if it improves both overlaps and cost or we go the temperature based acceptance if not.
- Temperature Start Centered: Same as Temperature Start but we position all nodes in the middle before starting, ignoring unpositioned list. We do so as to address the begin acceptance that is described in one of the next paragraphs
- Temperature Start Inverted Centered: Same as Temperature Start but we position all nodes in the middle before starting, ignoring unpositioned list. We do so as to address the begin acceptance that is described in one of the next paragraphs

The idea that came up to my mind is: if we keep ending in a local minimum and we reheat for x times it would maybe be better if we restart the run with a new starting solution, as it might be that to find a global minimum we need to make so many bad choices as to chnage the structure thata reheating to a small probability or just a reheat may not lead to enough chnages for us to find a good valley.

The Temperature Start diverges because the high temperature makes it so that the nodes are positioned on any nodes.

The Temperature Start Inverted converges more than the non-inverted counterpart but still has a lot of acceptance right away because of the fact that we position ndoes that probably have not a dependency between them so the cost is very reluctant to ramp up, and as such they "maintain" the solution cost at 0, which implies that those initial nodes are basically random positioned. We position 34 nodes right away (basically random) and the remaining 23 using a temperature based selection. The same way high initial temperature of the SA regular procedure makes the search diverge greatly and move back to costs of random solutions.

Either I see how it changes then start temperature coefficient is 1 or I will try to start SA from freezing temperatures.

As the 100 temperature makes the start solution diverge and 1 solution too I will benchmark:

- 1 temperature coefficient and initial strategies: Greedy, Temperature start, Temperature start inverted from center

I will also benchmark the restart strategy on all benchmarks.

I did a few benchmarks with high T (* 10 / 100) to show that for computed starts it diverges to random beginning

---

I printed more information about the DFG graph of both hotspot3D and cfd, which are similar in size, and close with difficulty, with hotspot being more easier than cfd but the closer one to it. I want to get as many information as possible as what makes cfd much more difficult than hotspot3D. I need to analyze them better.

---

Thought: It is not about the position relative to PEs but the position relative to other operations: the structure it is forming compared with other operations: Can we make a 3D image of how operations are positioned? Like a CGRA from top at `t = 0` and moving down with layers until the last clock time.

### 27.04.26

I started by implementing a refactor that can help speed and algorithm programming: we define two functions for each strategy to implement: `neighbour_sol_generator` and `undo_neighbour_sol_generator`.

Each routine is now also changed to always hold two solution at any given time: the best solution found and the current neighbor solutions denoted as `node_pe` and `curr_node_pe` respectively. Each time we operate over `node_pe`, in the case that aslo `pe_nodes` is needed, it is indeed compute each time a new solution is accepted.

It does create mutation complexity: we are always modifying the same DS in multiple region of the code, but it is the cost for less code and faster execution.

I did indeed applied the refactor, now the code is much lighter but the time did not change much, and it did not prove to be faster, for a recorded run it is actually slower of about 2s. But another benchmarks just improved from 210s to 110s, almost dividing the time. I think that overall the refactor is an improvment for the less complexity in the code structure, (probably) speed at cost of more mutation intricancy.

---

I did all greedy benchmarks.

---

I have written the analytics algorithm but I think the CSV with all algorithms in line is not going to make it, because it is very hard to read as we have a lot of different algorithm versions, up to 105.

I think I need to provide plots in someway that makes it for a better analysis, maybe plots that aggregate the same algortihm names, and similar such as the only reheating coeffcicient difference.

It would also be good to have a "top 10" algorithms were i display the top 10 out of all algorithms.

### 04.05.26

I finished implementing the analytics at a stable stage. While doing so I found that some of the strategies that have been implemented about lists have not been brought to the new benchmarking system, thus have not been saved in any CSV. Strategies like: MOST-CLOCK-DEPENDENCIES with all poisson types, WORST-POSITIONED-MOST-DEPENDENCIES-NODE with all poisson types and WORST-POSITIONED-NODE without poisson. Should I migrate them and run some benchmarks?

---

Looking at the newly collected data the greey starting strategy with temperature starting at 10 seems to me to be the current best startegy out of them all, so when considering this week start I will pick up the Greedy 10. I might consider doing more benchmarks of all startegies to get better throughtfull data for analytics.

---

I am not sure about the benchmarks we need to do: if the random start is random and different for both or it is random but the same for the two strategies. I did random and different.

---

I have finished implementing a stable version of analytics with a good chart structure that I think it is capable of handling well comparisons, and it is also easy to tune for additional strategies comparisons on the fly.

I started by diving strategies into different macro-groups:

- `classic-implementations`, which comprehends all initial and simple strategies, especially important for the `Cooling` and `Morpher` cooling routines. Here we find implementation of the slow reheating and the first impleemntation of the SMA.
- `list-selection`, which is the next startegies that have been implemented in chronological order comprehend all startegis that pick nodes not at random but from a predefined list with an heuristic to define its order, each list is then picked from using a poisson distribution, that is either fixed at a lambda, proportional to the size of the list or following the temperature (higher it is, the more lambda is towards lower scoring nodes). We found that they do not provide any advantage from the random node selection, thus have been discarded.
- `spike-temperature`, which introduce a complete reset reheating to initial temperature. This has been introduced from a paper Cristian suggested, the temperautre is reset following SMA. We found a clear improvement and discarded any slow reheating strategy we had.
- `probability-reheating-routine`, which modify the spike reheating to be spiked to a specific temperature which will give a desired probability of accepting different solutions. We found that indeed it is favourable to the search and that a compelte reheating to temperatures that given a complete accepance rate are not ideal.
- `precomputed-start`, which modify the starting solution of the search by having an heuristic precomputed startegy instead of having a completely random starting strategy. 
- `special-strategies`, which comprehends all strategies that do not fit directly in a macrocategory because of a peculiarity within its implementation.

From charts we can easily see that strategies within the `classic-implementations` do find solutions to majority of problems (except cfd, hotspot3D, some backdrop and some nw) but have their obstacles as all strategies do provide a solution in a considerably high amount of time.

List selection strategies have very few benchmarks and from those few we can clearly see that classic implementation already perform better, they are unable of finding solutions also to simple problems such as aes and basicmath, and do not find immediately a solution to the simplest problem of benchmarks: bitcount.

Spike temperature makes us able to have a jump in performance and solutions: we find solutions to all problems except for cfd, with morpher reset SMA and cooling reset SMA being very fast at finding those solutions. Clearly an improvment that has to be kept moving forward.

Probability reheating has a very messy chart because of the large quantities of elements within it, we can have a better view of those startegies looking at some custom macro groups we build: `best-cost-different-reheating-probability`, `greedy-start-T-1-10-100`, `best-cost-T-10-different-reheating-probability`, `best-cost-T-1-10-100`, `best-cost-size-100-different-reheating-probability`,`learned-reheating-probability` and `fixed-p-reheating-probability`.
`fixed-p-reheating-probability` has a best time for reheating to `0.55`, with a time of around 80s, `learned-reheating-probability` blows it with reheating to `0.25` with a time of 32s, but bear in mind that those run may be greatly affected by luck, as I think are around 2 for those benchmarks, definitely not statistically significant. While `best-cost-T-10-different-reheating-probability` has much more stable times and rpovides best ones for rehating of `0.2` and `0.25` or `0.55` and `0.6`. We also have `best-cost-size-100-different-reheating-probability` that has much more balanced times providing their best of around 200/300s at probabilities `0.2` and `0.3` or `0.55` and `0.65`. Lastly `best-cost-T-1-10-100` compares all `T` coefficients 1, 10 and 100 at a parity of probability reheating: `0.45` and it is very hard to find the best visually as they all have their advantages and disadvantages benchmarks but overall coefficient of 1 seems to work best. Another entry is `best-cost-different-reheating-probability` which is still reheating to a fixed probbaility but with the hold fixed starting temperature of size times 100, it still provides good times, especially for a reheating probability of `0.25`.
Overall the probbaility reheating is a very good improvement over the spike reheating and deduces the speed of all benchmarks, especially the one for the hotspot3D. We still have some uncertanty of the robability reheating it is best to use, but a prbability in range `0.2-0.3` and `0.45-0.65` is a good choice. Also `greedy-start-T-1-10-100`, is a colelction of greedy starting strategies to change only from intial T coefficient, and similarly to the best cost macro group there is not a single strategy that takes the lead, but it seems to me that the 10 is better as for result in hotspot3D and the vicinity with the best strategy for benchmarks that does not win.

Precomputed start does provide an improvment over times, especially on smaller benchmarks, notably there are the greedy strategies but I think that the temperature start inverted does a better job compared to greedy strategies. Yet, we need to say that the computation time of the starting solution is not comprehended in benchmarks times, as when it was random it is was also this way, as also the simulated annealing algorithm has a requirement of initial solution to start running.
Definitely an improvement to be considered and added to future strategies.

Special strategies are all standalone strategies that for a "special" characteristic do not fit in any other macro-groups, these are usually more complex to architect. Right now we only have the restart startegy, which takes count of the last plateau cost and if it is stuck in a same or worse cost plateau for more thna three times in a row it will restart from a different starting solution, computed using the inverted centred temperature startegy to compute an initial solution. It uses random start as first initial solution. It is a very good strategy that does not come short from any other startegy, but it has been thought as being able to solve the hardest problem we currently have: `cfd`. It was not succesfull in solving `cfd`.


Notably for this week we wanted to benchmark greedy and random start with different starting solutions as we thought that greedy may be the best overall strategy we found so far. Results can be seen by the macro group collection `greedy-vs-random-start-T-10`. The benchmark shows that greedy is slightly better then random start, with median having greedy to be better than random in the hotspot3D benchmark.


Then I added a representative macro group that does show the evolution from strategy to strategy, I handy picked each strategy to be one of the most representative per category and it is at the dge of the messiness, but clearly shows what we just stated: probability reheating and greedy are very good improvments from basic classic strategies.


Then I also added top macrogroups:

- manual top, which I picked strategies I think are the best
- computed top, in which we caompute using weights multiplied to times, to enhance the importance of hard benchmarks to see which strategies are the best, for both mean and median times, to spot if numbers say something that I couldn't see by myself. They are computed over all benchmarkes strategies.

For the top manual selection I have:

GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP,
GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP,
GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP,
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP

Which includes our bias towards greedy.

For the computed ones we get, mean:

TEMPERATURE-START-INVERTED_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP,

median:

TEMPERATURE-START-INVERTED_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP,
RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP,

where both have the list from best to worst. These results put at the top a computed start strategy: the temperature start inverted instead of greedy, next the random start cooling and following the greedy, which shows to be very solid, ednig with the special restart. One thing that I would like to point out is that all strategies have the fixed probability reheating to 0.45, and a startign coefficient of 1 isntead of 10 or 100.
This list confirms that the rehating to probability, and the computed start do have an advantage and it is probably better to have a T coefficient of 1.

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
[acceptance_rate of morpher](https://deepwiki.com/search/devin-look-at-the-simulated-an_dd173aac-0ddd-40c1-96cd-d53915e91121) 03.03.26 at 14:05

[overflow of mat.exp](https://www.py4u.org/blog/python-overflowerror-math-range-error/#why-mathexp-throws-error)

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

### Testing

- [] Add testing suite

- [] Add tests to make sure that computation of pe distances is correct
- [] Add tests to make that that computation of starting solutions coincides

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

[x] Add more data and insights do the benchmark
- [x] We add the number of nodes that are positioned correctly and the number of nodes that are not, along a lists of the type: \[source-destination, ..\] for incorrect positioned nodes
- [x] We add each node and relative the cost of the solution that they have for the current strategy, i.e. the distance that they have between them, an array of the type: \[source-destination: cost\]
- [x] Added average routine times
- [x] Added `--dump` flag that dumps the data of the run in a CSV along with ictures of the run

[] Improve and enhance plots
- [x] Added single temperature / cost chart
- [x] Added CGRA mappings plot
- [x] Moved temperature to probability
- [x] Divided the run in three plotted linearly those three temperature / cost tranches
- [x] Added window averaged plot
- [x] Added run plots: each spike to colling window is divided and plotted from the start on the same plot

- [] Add visits heatmap plot
- [] Add problem cost function valleys (contour plot for multidimensional problem)

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
- [x] Completely random on available positions and not (swapping)
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

   > Right now we are not sure that the computation is going to terminate even if we know that a solution exist as for more intensive problems, such as the particlefilter, we might need more time than given. Morpher uses the acceptance probability to decide how to decrease the temperature, we could look at first, simply to keep temperature a little bit hot to keep it running until we make considerable improvements between iterations.

- [x] Simplify SMA computation: Just an average of the last x results

- [x] Modify the temperature to be detached from the iteration cycle, make the algorithm iterate a fixed amount of time and make it 

- [x] Make the temperature follow the cost like a MA: make it decrease a little if the last iterations have been good iterations (we improved) or decrease a little if the opposite. Maybe may be good to make temperature start at the cost and then diminish as we approach 0 and freeze at 0 (integers), since it is not a problem to iterate for long, not as a float 100-0.0001.

- [x] Add reset temperature completely to the beginning when stuck to random and start benchmarking

- [] Implement a Hybrid strategy: As the temperature gets colder the startegy has more chanche, or switches copletely, to a more heuristic startegy than before, so that it points more to the refinment of the solution as the temperature decreases.

- [x] Apply the temperature in three different ways to the algorithm:
   1. The temperature is used only to decide whether we pick or not a worst solution (hill climb)
   2. The temperature is used also to change the distribution of Poisson's lambda, the greather the temperature the greater the lambda, the less heuristic the search will be
   3. The temperature is used to change the distribution peak of a Gaussian distribution (variance and probability), the more high it is the less heuristic the search will be

[x] Implement a reset to temperatures based on a probability that we choose
   - [x] Dynamic: use use average cost of items we have seen this run: alrready makes different plateau levels increase differently the temperature, the worse the plateau is, the more the temperature will be.
   - [x] Use starting solution cost as baseline $\Delta c$
   - [x] Use maximum worst CGRA solution cost as baseline $\Delta c$
   - [x] Use the current best solution cost as $\Delta c$
   - [x] Use a learned table: while first descent collect temperatures at which a given acceptance probability is given for the problem
   - [x] Use a fixed probability routine: Define the acceptance probability independent from temperature and move it using fixed items intervals

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
aggiustare temperatura che segue il costo come detto (un'occhiata all'implementazione di morpher)

### 09.03.26

Idee per la nuova temperatura:

- Temperatura ugualle alla MA +- epsilon random dipendente dal costo: Costo basso più epsilon è basso 
- Fai seguire la temperatura con la media mobile (no freezing temperature), e nel riscaldamento al massimo più di 50% di differenza dalla MA
- MA +=5% e vedi come si comporta

Prima: aggiungere extra dati ai risultati, poi fare benchmarkings:

1. benchmarking con morpher (da random) e no riscaldamento + random: almeno uno per algo e dimensione

2. benchmarking con morpher (da random) e riscaldamento attivato con MA (vecchio: Temperatura usa MA per decidere quando scaldare, modifica: riscalda molto più lentamente) + random: almeno uno per algo e dimensione

3. benchmarking con temperatura MA (nuovo) + random: almeno uno per algo e dimensione, provare prima run indipendenti per trovare il migliore, poi benchmarks

benchmarking con morpher (da random) e riscaldamento con Altre idee (pensa a come relazionare temperatura con MA/Costo) per temperatura + random: almeno uno per algo e dimensione

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

controlla cooling random node tempi e iterazioni: sono sbagliate, tanto tempo per un po' poche iterazioni:

COOLING_RANDOM-NODE	4000.7517680000000	26	2945	47
COOLING_RANDOM-NODE-WITH-SWAP	4000.203499	20	3284	47

Come mai con 300 iterazioni in più ha lo stesso tempo? Controlla

Come mai nelle temperatura riscaldata da SMA scende così in fretta dopo essere riscaldata e non ad un rate simile visibilmente come il raffreddamento iniziale?

benchmark con seed fisso
.. poi senza, per ora fissiamo: generare solo la configurazione iniziale uguale, poi le scelte dell'arlgoritmo indipendenti dal seed, true random (priorità alta)

vedere alcuni benchmark che cosa succede entrambi con lo stesso seed (priorità bassa)

particlefilter	38	10	10	MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE-WITH-SWAP	754.0652670000000	0	18568
particlefilter	38	10	10	MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE-WITH-SWAP	70.64206300000000	0	1021
particlefilter	38	20	20	MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE	4000.0007950000000	104	16547

come mai quasi a parità di iterazioni la 20x20 impiega molto di piu della 10x10? La size non dovrebbe influire così tanto sul tempo

Controllare che effettivamnte le itrerazioni della temperatura sono le stesse anche più in la nelle interazioni: scala logaritmica non le fa vedere con la stessa distanza in asse x

cooling: temperatura iniziale basata sull'acceptance rate del problema, circa un 30% di acceptance rate all'inizio, leggere il paper

controllare discordanze tempi: da fare subito

csv collezzionare float al secondo decimale e aggiusta formatting costi degli edges

check that temperature cooling has same iterations as start cooling: graph doesn't let it show

long lassting task to perfect over time:
maybe adjust better graph visual also for latter iterations

- morpher + spike: 100 MA lenta con: reset SMA e senza reset
- cooling + spike: +100 MA e 10 * numero di nodi non 100
- seed: per tutte le benchmark fixare le configurazioni iniziali

- SMA 100 è a +-2% del costo per tot iterazioni o tempo

mantenere una lista globale ogni nodo così che puoi selezionare il pe in cui posizionarsi: non cambia la configurazione ma risulta più semplice l'implementazione

controlla configurazioni iniziali siano uguali: potrebbe non essere così: it is, i'll leave the print of the solution to file, and cost to benchamrk so that if it is equal for each benchmark run we can be sure they are the same

+ + benchmarks and to implement
- tavola delle configurazioni visitate: senza moving averages + cooling raffreddamento e spike and add plot: heatmap of visits + cost (do first only visits)

rifare benchmarks con la temperatura settata in modod che parta a 30% di acceptance rate: aggiungi alla cooling schedule più promettente

### 23.03.26

Added to 17.03.26

### 30.03.26

Fixed temperature reset percentage: many fixed level runs, save id with percentage reset level: from 5% up to 80% with step of 5%
colleziona la % di selezionare una nuova soluzione se peggiore: l'intuizione sarebbe di arrivare ad una temperatura che corrisponda ad una % di probabilità di accettare la soluzione, sarebbe l'ideale trovare una formula che ci possa dare la % esatta di probabilità di accettare una nuova soluzione: se la run arrivata ad un plateau sarebbe l'ideale tornare al 30% di probabilità di acceptance, girando la formula base in modo che dia la temperatura $T$ a cui settare.
Facciamo benchmarks con stesso problema, stesso algoritmo e stesso seed di esecuzione, cambia solo la % a cui faccciamo il reset della temperatura.
collezzioniamo i dati di ogni run separatamente in un nuovo file, così possiamo fare plot uno sull'altro delle varie run
Enhance benchmarking so that we can collect separated data runs


Dynamic temperature reset percentage: keep best promising level from previous benchmarks and apply dynamicity: should be addressed by how we manage the temperature right now 5% for smaller plateau, 50% for equal or larger: keep as subsituted by first fixed temperature benchmarks because of dynamicity.
spike: in %: se plateau è più alto di quello di prima più spike, meno o uguale un po' di meno: così che accetti meno soluzioni se migliora e il costo sale di meno: non facciamola uguale alla temperatura iniziale. Esperimenti con: uno con il 5% uno con il 50%. Potrebbe essere già implementato dallo spike con dynamic


Temperatura iniziale correlata al problema
Usiamo le dimensioni del grafo del problema per la temperatura e non la size: * 1/10/100: su un singolo benchmark in cui trova la soluzione guardare quello che ci mette di meno e come si comportanto a 3 soluzioni iniziali: per le 3 temperature 3 set ognuna con una configurazione iniziale: così vediamo come si comportano

9 esperimenti per ogni t * 100/10/1 con 3 schedule diversi:

- 3 setssa configurazione iniziale e stesso model number
- 3 con lo stesso model number (4) e configurazioni iniziali diverse
- 3 mix: diverse schedules (mode_number) e configurazioni iniziali

- stessa schedule == stesso numero in break a `model_number` (controlla che il seed che usa per la generazione sia fissato e non influisca) per schedule diverse cambia `if model_number == 4`: # 4 a linea 928 a 1 di `monomap.py`
- 2/3 benchmarks almeno con un problema da 20 nodi per benchmark, cerchiamo un problema come particlefilter in cui a volte trovo al soluzione a volte no


Cambiare la scala della temperatura alla probabilità: Passiamo da temperatura a % di probabilità di selezionare una soluzione peggiore, così che per noi abbia più senso

### 13.04.26

Check that the start of the two following benchmarks:

/Users/eliariva/Documents/Puddi/USI/BachelorProject/MonoMap/benchmarks-results/hotspot3D/20x20/COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4/2026-04-11T13-25-29

/Users/eliariva/Documents/Puddi/USI/BachelorProject/MonoMap/benchmarks-results/hotspot3D/20x20/COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4/2026-04-10T21-43-01

Have the same starting cost as the graph does not make it seem like it is.


Another variation of temperature reheating: Per scaldare come $\Delta c$ prendiamo l'ultimo jump verso il basso: l'ultimo $\Delta c$ negativo che ho trovato come $\Delta C$ per calcolare il reheating della temperatura ad una probabilità

Implement hold methods of reheating and make a few benchmarks pictures to then show to Cristian, I should have collected them in benchmakrs and to show them to him last week, should have been more proper. We do:

- learned rehating
- average solution cost
- new implementation: last delta to plateau

run only on hotspot3D

1/10/100 * nodes in dfg as initial temperature: do all benchmarks for each coefficient with best strategy
Testare più cfd problemi per vedere se falliscono proprio tutti con la strategia migliore


Start solution calculated, make the three algorithms:

- Start from initial overlapping PEs and select random position in direct neighbor to open them aout (Should be already implemented)
- From greedy cost function: I have two functions costs, which are overlapping costs and regular solutio cost, I place all operaitons on the same central PE and for each I select the next PE that minimizes both function, once the overlapping cost is zero we start the regulat SA
- From outside list: make a PE list that are not yet positioned, positions one at random, then postion others at random in this way: if cost remains the same (best position) position it, else we accept it folllowing a temperature probability, and i continue until all PEs are placed, if so we start regular SA
- Extra version: use the temperature SA into a current SA routine, then when we reach the same plateau or worse after x (maybe 2) reheats we rebuild the solution from the beginning and repeat SA

and benchmark three starting solution with all sets


Use networkx to print statistics about the graph and understand what it changes between cfd and hotspot3D graph: identify which nodes are a problem and which characteristics make them a problem.

### 27.04.26

Benchmark Greedy also for \*100 for initial T (\* 1/10/100): all benchmarks, we skip cfd

Controllare qualitativamente gli algoritmi tra di loro per capire quale sia il migliore (post processing dei CSV in un unico CSV con visuale orizzontale: a sinistra il nome degli algoritmi e strategie sulle colonne, poi entries i tempi: tutti i CSV che abbiamo collezionato)
una tabella con average e un'altra, separate da una riga bianca, con mediano delle run se c'è senno solo average

### 04.05.26

Cercaro macrogruppi all'interno delle benchmarks e suddividere così i dati, poi estrarne i più rappresentativi e/o migliori
Scelta manuale dei gruppi anche potrebbe andare bene
magari fare più confronti tra categorie
5/6 categorie più rappresentative

benchmarks con configurazioni inziali diverse con random e con greedy e cercare il mediano di 10+ run quale sia migliore

### Extra

+ + extra: anything that follows are extra ideas I got while working
+ heatmap like zones of probability: each node has a heatmap probability of the CGRA based on where its dependencies are positioned withoin the CGRA, the more dependencies are in a zone the more probable such PE will be to be picked when such operation is moved

+ maybe if time look at how you can draw the space cost in a 3D plot (3D Contour) with valleys even if it is a more than a 3D space

+ Fixed temperature stages
+ do benchmarks also for fixed temperature ranges, like: 100 items at 0.5 accept rate, 300 at 0.2, 750 at 0.05 etc, set P as fixed values independent from cost, or T

+ what about going to next temperature level when the acceptance rate arrives to be equal to the temperature? the meaning is to arrive to a point where the expected probability of selection is met, which may be correlated to the temperature level reaching the correct plateau state, equilibrium for that temperature as Timberwolf suggested

+ Addition to the random movement algorithm: When you select a node to be moved check that it does not have a loop dependency, if so move it else take the starting dependency node and move it with the current node so that the dependency is respected: make the new placement and for the loop dependency node take the valid nodes for dependency at random select one and place it there or swap it with the ndoe that is already there. Then test it only on CFD problem.
- Remember to make it two way, if one node gives or receives a loop dependency with another one it has to follow the couple swap

+ dopo
+ nodi con tante out dependencies (solo su nodi con out degree > 2/3 o numero di figli > connettivity degree): i fligli possono essere posizionati solo in un intorno del nodo parente: uno in cui random e prendere un nodo random, guardare il padre e posizionare negli hop one del padre se sono pieni: facciamo lo swap con un nodo lì: benchmarks cfd hotspot3D, terzo con più nodi

+ Follows the heatmap of probabilities: look at all parents of a single node, and draw the heatmap as follows: start by colleting parent positions, same position has 1 probability, 1 hop has 1/2, two hops have 1/4 and so on for each parent, then you normalize probabilities. Implement also the staretgy

+ pensare: "freezare" le connessioni dei nodi posizionati bene: parente e figli (da considerare)
