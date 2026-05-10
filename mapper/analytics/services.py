
import numpy as np

from analytics.models import Row, AnalyticsRepresentativesColumn

def compute_similar_algorithms(rows: list[Row]) -> list[list[Row]]:
    # we compute similarity
    # straight away: similarity is is the sa algorithm name
    # changes only by a few characters

    # collect different algorithm types
    different_sa_types: list[str] = []
    for r in rows:
        if r.sa_algorithm_type not in different_sa_types:
            different_sa_types.append(r.sa_algorithm_type)

    # compute similarities
    similar_sa_types: list[list[str]] = []
    for r in rows:
        type = r.sa_algorithm_type

        if len(similar_sa_types) == 0:
            similar_sa_types.append([type])
            continue

        for i in range(len(similar_sa_types)):
            # if it is in the list it will show up in the similar list
            # hard to find the same type in two different lists with a
            # good similarity policy
            if type in similar_sa_types[i]:
                break

            # check similarity
            for t in similar_sa_types[i]:
                distance = 0

                # compute character distance
                for j in range(len(t)):
                    if len(type) <= j:
                        break
                    elif len(t) <= j:
                        break
                    elif t[j] != type[j]:
                        distance += 1

                # if similar
                if distance < 4:
                    similar_sa_types[i].append(type)

    # filter by similarities
    similar_sa_types_rows: list[list[Row]] = [[] for _ in range(len(similar_sa_types))]
    for r in rows:
        for i, types in enumerate(similar_sa_types_rows):
            if r.sa_algorithm_type in types:
                similar_sa_types_rows[i].append(r)
    return similar_sa_types_rows


def return_top_algorithms(
        sa_algorithm_type_algorithm_times: dict[str, dict[str, list[float]]],
        BENCHMARKS: list[str], WEIGHTS: dict[str, int],
        MANUAL_MACRO_GROUPS: dict[str, list[str]],
        by_mean = True,
        top: int = 4
    ) -> dict[str, dict[str, AnalyticsRepresentativesColumn]]:
    # We compute "best" by the smallest weighted average of times, weighted
    # in a way that the higher node count is the more weight such time has (not totally representative tho)

    sa_algos_analytics: list[AnalyticsRepresentativesColumn] = []
    for sa_algo, algorithm_times in sa_algorithm_type_algorithm_times.items():
        algorithms_mean: dict[str, float] = {}
        algorithms_median: dict[str, float] = {}

        # which macrogroup?
        macro_group: str | None = None
        for mmg, sa_algos in MANUAL_MACRO_GROUPS.items():
            if sa_algo in sa_algos:
                macro_group = mmg

        for algo, times in algorithm_times.items():
            np_times = np.array(times)
            mean = np_times.mean()

            np_times.sort()
            ntimes = len(np_times)
            median = np_times[ntimes // 2] if ntimes % 2 != 0 else (np_times[ntimes // 2 - 1] + np_times[ntimes // 2]) / 2

            algorithms_mean[algo] = mean
            algorithms_median[algo] = median

        # all analytics
        sa_algos_analytics.append(AnalyticsRepresentativesColumn(sa_algo, macro_group, algorithms_mean, algorithms_median))

    # still, how can i evaluate algos that do not have all benchmarks within it?
    # add a malus to balance them out?
    # disqualify them? such as skipping them, for now yes
    # build a mapping of sa_algo -> benchmark -> rows
    # for sure we do not consider cfd, as all of them are timining out
    # and i dont want to disqualify algos that have not benchmarked it
    # filter out missing benchmarks
    to_delete_indexes: list[AnalyticsRepresentativesColumn] = []
    for algo in sa_algos_analytics:
        if len(algo.algorithm_mean_time) < len(BENCHMARKS) or (len(algo.algorithm_mean_time) == len(BENCHMARKS) - 1 and "cfd" not in algo.algorithm_mean_time):
            to_delete_indexes.append(algo)
    
    for arc in to_delete_indexes:
        sa_algos_analytics.remove(arc)

    # compute coefficients (index, score)
    top_sa_algorithms: list[list[int, int]] = []
    for i, sa_algo in enumerate(sa_algos_analytics):
        top_sa_algorithms.append([i, 0])

        for benchmark in sa_algo.algorithm_mean_time:
            if benchmark == "cfd":
                continue

            if by_mean:
                top_sa_algorithms[-1][1] += sa_algo.algorithm_mean_time[benchmark] * WEIGHTS[benchmark]
            else:
                top_sa_algorithms[-1][1] += sa_algo.algorithm_median_time[benchmark] * WEIGHTS[benchmark]

    # sort by computed value
    top_sa_algorithms.sort(key=lambda x: x[1])

    # tabe best
    top_sa_algorithms_rows: dict[str, dict[str, AnalyticsRepresentativesColumn]] = {}
    for t in range(top):
        if len(top_sa_algorithms) <= t:
            break

        sa_algo = top_sa_algorithms[t]
        algo_analytics_rep: AnalyticsRepresentativesColumn = sa_algos_analytics[sa_algo[0]]
        algo_analytics_rep.id = str(t + 1)

        if algo_analytics_rep.macro_group not in top_sa_algorithms_rows:
            top_sa_algorithms_rows[algo_analytics_rep.macro_group] = {}

        top_sa_algorithms_rows[
            algo_analytics_rep.macro_group
        ][
            algo_analytics_rep.sa_algorithm_type
        ] = algo_analytics_rep
    return top_sa_algorithms_rows
