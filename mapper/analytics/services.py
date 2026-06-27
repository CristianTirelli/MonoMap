
import numpy as np

from analytics.models import Row, AnalyticsRepresentativesColumn

# TODO should make class or constants section
BENCHMARKS = [
    "aes",
    "backprop",
    "basicmath",
    "bitcount",
    "cfd",
    "crc32",
    "fft",
    "gsm",
    "heartwall",
    "hotspot3D",
    "lud",
    "nw",
    "particlefilter",
    "sha1",
    "sha2",
    "stringsearch",
    "susan",
]

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
        sa_algorithm_type_algorithm_rows: dict[str, dict[str, list[Row]]],
        BENCHMARKS: list[str], WEIGHTS: dict[str, int],
        MANUAL_MACRO_GROUPS: dict[str, list[str]],
        by_mean = True,
        top: int = 4
    ) -> dict[str, dict[str, AnalyticsRepresentativesColumn]]:
    # We compute "best" by the smallest weighted average of times, weighted
    # in a way that the higher node count is the more weight such time has (not totally representative tho)

    sa_algos_analytics: list[AnalyticsRepresentativesColumn] = []
    for sa_algo, algorithm_rows in sa_algorithm_type_algorithm_rows.items():
        algorithms_mean_timeout: dict[str, bool] = {}
        algorithms_median_timeout: dict[str, bool] = {}

        algorithms_mean: dict[str, float] = {}
        algorithms_median: dict[str, float] = {}

        algorithms_mean_norm: dict[str, float] = {}
        algorithms_median_norm: dict[str, float] = {}

        algorithms_mean_iteration: dict[str, float | None] = {}
        algorithms_median_iteration: dict[str, float | None] = {}

        # which macrogroup?
        macro_group: str | None = None
        for mmg, sa_algos in MANUAL_MACRO_GROUPS.items():
            if sa_algo in sa_algos:
                macro_group = mmg

        for algo, rows in algorithm_rows.items():
            np_iterations = np.array([r.iterations for r in rows])
            mean_iterations = np_iterations.mean()

            np_iterations.sort()
            niterations = len(np_iterations)
            median_iterations = np_iterations[niterations // 2] if niterations % 2 != 0 else (np_iterations[niterations // 2 - 1] + np_iterations[niterations // 2]) / 2

            np_times = np.array([r.time_seconds for r in rows])
            mean = np_times.mean()

            np_times.sort()
            ntimes = len(np_times)
            median = np_times[ntimes // 2] if ntimes % 2 != 0 else (np_times[ntimes // 2 - 1] + np_times[ntimes // 2]) / 2

            algorithms_mean_timeout[algo] = np.all([r.is_timeout for r in rows])
            algorithms_median_timeout[algo] = rows[ntimes // 2].is_timeout if ntimes % 2 != 0 else rows[ntimes // 2 - 1].is_timeout and rows[ntimes // 2].is_timeout

            algorithms_mean[algo] = mean
            algorithms_median[algo] = median

            algorithms_mean_norm[algo] = mean / np_times.max()
            algorithms_median_norm[algo] = median / np_times.max()

            algorithms_mean_iteration[algo] = mean_iterations / np_iterations.max()
            algorithms_median_iteration[algo] = median_iterations / np_iterations.max()

        # all analytics
        sa_algos_analytics.append(AnalyticsRepresentativesColumn(
            sa_algo,
            macro_group,
            algorithms_mean_timeout,
            algorithms_median_timeout,
            algorithms_mean,
            algorithms_median,
            algorithms_mean_norm,
            algorithms_median_norm,
            algorithms_mean_iteration,
            algorithms_median_iteration
        ))

        # if some benchmark missing add nones
        for b in BENCHMARKS:
            if b not in algorithms_mean_timeout:
                algorithms_mean_timeout[b] = None
            if b not in algorithms_median_timeout:
                algorithms_median_timeout[b] = None
            if b not in algorithms_mean:
                algorithms_mean[b] = None
            if b not in algorithms_median:
                algorithms_mean[b] = None
            if b not in algorithms_mean_iteration:
                algorithms_mean_iteration[b] = None
            if b not in algorithms_median_iteration:
                algorithms_median_iteration[b] = None

    # still, how can i evaluate algos that do not have all benchmarks within it?
    # add a malus to balance them out?
    # disqualify them? such as skipping them, for now yes
    # build a mapping of sa_algo -> benchmark -> rows
    # for sure we do not consider cfd, as all of them are timining out
    # and i dont want to disqualify algos that have not benchmarked it
    # filter out missing benchmarks
    to_delete_indexes: list[AnalyticsRepresentativesColumn] = []
    for algo in sa_algos_analytics:
        algorithm_mean_time_nones = 0
        for time in algo.algorithm_mean_time.values():
            if not time:
                algorithm_mean_time_nones += 1

        not_none_algorithm_mean_times = len(algo.algorithm_mean_time) - algorithm_mean_time_nones

        if not_none_algorithm_mean_times < len(BENCHMARKS) and algo.algorithm_mean_time["cfd"]:
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
