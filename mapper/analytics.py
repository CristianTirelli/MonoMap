from datetime import datetime
from pathlib import Path
import numpy as np

from analytics.csv import AnalyticsColumn, load_all_rows, write_analytics_csv
from analytics.models import AnalyticsRepresentativesColumn, Row
from analytics.services import return_top_algorithms
from analytics.plots import plot_mean_median_barcharts

BASE_PATH: Path = Path(f"analytics-results/{datetime.now().strftime("%d-%m-%y")}")
REPRESENTATIVES_PATH: Path = Path("representatives")
TOP_PATH: Path = Path("top")
TOP_COMPUTED_PATH: Path = Path("top-computed")

CSV_BASE_PATH: Path = Path("benchmarks-results")
CSV_NAME: str = "data-benchmarks.csv"

# benchamrks
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


# Remember: only one per macrogroup
MANUAL_MACRO_GROUPS = {
    "classic-implementations": [
        # morpher
        "MORPHER_RANDOM-NODE",

        # colling
        "COOLING_RANDOM-NODE",
        "COOLING_RANDOM-NODE-WITH-SWAP",

        # morpher and sma
        "TEMPERATURE-SMA_RANDOM-NODE",
        "MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE",
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_RANDOM-NODE",

        # morpher and sma with swap
        "MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE-WITH-SWAP",
    ],

    "list-selection": [
        # TODO check i did much more lists strategies... but benchmarks are not there
        # Some of them have been developed before the benchmarking system has been set up
        # and seen their performance they havent been migrated to the new system.
        # Should I migrate them and benchmark them again?


        # poisson distribution fixed
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # poisson proportional to operation nodes array
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-PROPORTIONAL-0.25",

        # poisson worst positioned
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-PROPORTIONAL-TEMPERATURE",
    ],

    "spike-temperature": [
        # t reset
        "MORPHER-RESET-SMA_RANDOM-NODE",

        # t and sma reset
        "MORPHER-RESET-SMA-RESET-SMA_RANDOM-NODE",
        "MORPHER-RESET-SMA-RESET-SMA_RANDOM-NODE-WITH-SWAP",

        # random node
        "COOLING-RESET_RANDOM-NODE",

        # new routine implementation
        "COOLING-RESET_RANDOM-NODE-WITH-SWAP-NEW-ROUTINE",
        "COOLING-RESET_RANDOM-NODE-WITH-SWAP-NEW-ROUTINE-FASTER-COPY",
    ],

    "precomputed-start": [
        # t start centered
        "TEMPERATURE-START-CENTERED_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",

        # t start inverted
        "TEMPERATURE-START-INVERTED_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
        "TEMPERATURE-START-INVERTED_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",

        # t start
        "TEMPERATURE-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
        "TEMPERATURE-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",

        # greedy start 1, 10 and 100
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "probability-reheating-routine": [
        # last delta
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-LAST-DELTA-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",

        # learned
        "RANDOM-START_COOLING-RESET-TO-0-2-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-25-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-3-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-35-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-4-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-5-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-55-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-6-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",

        # best cost, random start and size * 100 start T
        "COOLING-RESET-TO-0-05-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-1-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-15-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-2-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-25-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-3-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-35-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-4-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-55-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-6-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-65-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-7-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-75-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-8-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",

        # best cost 1
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",

        # best cost 1 special benchmarks: 3 same init and model, 3 different model, 3 different init and model all same and different run seed
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8-SAME-RUN",

        # best cost 10
        "RANDOM-START_COOLING-RESET-TO-0-2-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-25-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-3-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-35-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-4-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-55-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-6-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",

        # best cost 10 special benchmarks: 3 same init and model, 3 different model, 3 different init and model all same and different run seed
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8-SAME-RUN",

        # best cost 100
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",

        # best cost 100 special benchmarks: 3 same init and model, 3 different model, 3 different init and model all same and different run seed
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-SAME-INIT-CONF-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-SAME-MODEL-NUMBER-4-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-2-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-6-SAME-RUN",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP-DIFF-INIT-CONF-DIFF-MODEL-NUMBER-8-SAME-RUN",

        # fixed p
        "RANDOM-START_FIXED-P-RESET-TO-0-25_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-2_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-35_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-3_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-45_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-4_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-55_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-5_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-6_RANDOM-NODE-WITH-SWAP",
    ],

    "special-strategies": [
        # restart
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP"
    ]
}

# Remember: only one per macrogroup
MANUAL_REPRESENTATIVES = {
    "classic-implementations": [
        # morpher
        "MORPHER_RANDOM-NODE",
        # morpher and sma
        "TEMPERATURE-SMA_RANDOM-NODE",
    ],

    "spike-temperature": [
        # random node
        "COOLING-RESET_RANDOM-NODE",
    ],

    "list-selection": [
        # poisson distribution fixed
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-FIXED-2",
    ],

    "precomputed-start": [
        # greedy start 1, 10 and 100
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "probability-reheating-routine": [
        # best cost 1
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",

        # best cost 10
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",

        # best cost 100
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "special-strategies": [
        # restart
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP"
    ]
}

# Remember: only one per macrogroup
MANUAL_TOP = {
    "precomputed-start": [
        # greedy start 1, 10 and 100
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "probability-reheating-routine": [
        # best cost 10
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],

    "special-strategies": [
        # restart
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1-RESTART_RANDOM-NODE-WITH-SWAP"
    ]
}


ALL_GROUPS_ALGO_UNIQUES = {
    "MANUAL_MACRO_GROUPS": MANUAL_MACRO_GROUPS,
    "MANUAL_REPRESENTATIVES": MANUAL_REPRESENTATIVES,
    "MANUAL_TOP": MANUAL_TOP
}


# Can be more than one per group
MANUAL_CUSTOM_MACRO_GROUPS = {
    # "id-group" : [
    #     # algo in groups
    # ]

    "greedy-start-T-1-10-100": [
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "greedy-vs-random-start-T-10": [
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],

    "best-cost-size-100-different-reheating-probability": [
        "COOLING-RESET-TO-0-05-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-1-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-15-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-2-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-25-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-3-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-35-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-4-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-55-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-6-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-65-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-7-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-75-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
        "COOLING-RESET-TO-0-8-DYNAMIC-BEST-COST_RANDOM-NODE-WITH-SWAP",
    ],

    "best-cost-T-1-10-100": [
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-1_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-100_RANDOM-NODE-WITH-SWAP",
    ],

    "best-cost-T-10-different-reheating-probability": [
        "RANDOM-START_COOLING-RESET-TO-0-2-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-25-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-3-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-35-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-4-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-5-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-55-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-6-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],

    "learned-reheating-probability": [
        "RANDOM-START_COOLING-RESET-TO-0-2-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-25-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-3-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-35-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-4-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-5-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-55-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_COOLING-RESET-TO-0-6-DYNAMIC-LEARNED-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],

    "fixed-p-reheating-probability": [
        "RANDOM-START_FIXED-P-RESET-TO-0-25_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-2_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-35_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-3_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-45_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-4_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-55_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-5_RANDOM-NODE-WITH-SWAP",
        "RANDOM-START_FIXED-P-RESET-TO-0-6_RANDOM-NODE-WITH-SWAP",
    ]
}


def append_nested(
    container: dict[str, dict[str, list]],
    outer_key: str,
    inner_key: str,
    value,
):
    if outer_key not in container:
        container[outer_key] = {}

    if inner_key not in container[outer_key]:
        container[outer_key][inner_key] = []

    container[outer_key][inner_key].append(value)


def are_strategies_unique(GROUPS: dict[str, list[str]]) -> bool:
    # build a list of all entries
    algos: list[str] = []
    for g in GROUPS:
        algos += g

    # check for duplicate: naive..
    for i in range(len(algos)):
        for j in range(len(algos)):
            if i != j and algos[i] == algos[j]:
                return True
    return False



if __name__ == "__main__":
    # ensure uniqueness
    for k, v in ALL_GROUPS_ALGO_UNIQUES.items():
        if not are_strategies_unique(v):
            print(f"Unique algorithms group doesn't have unique algorithms: {k}")
            exit(1)



    ########### read all CSV files from /benchmark-results
    rows: list[Row] = load_all_rows(CSV_BASE_PATH, CSV_NAME)
    print(f"We have: {len(rows)} rows")
    ###########



    ########### Collecting by SA_ALGO all its ALGORITHM BENCHMARKS all their 20x20 TIMES
    sa_algorithm_type_algorithm_times: dict[str, dict[str, list[float]]] = {}
    skip_sizes = [2, 5, 10]
    for r in rows:
        if r.size_x in skip_sizes and r.size_y in skip_sizes:
            continue

        if r.sa_algorithm_type not in sa_algorithm_type_algorithm_times:
            sa_algorithm_type_algorithm_times[r.sa_algorithm_type] = {}
            sa_algorithm_type_algorithm_times[r.sa_algorithm_type][r.algorithm] = [r.time_seconds]
        else:
            if r.algorithm not in sa_algorithm_type_algorithm_times[r.sa_algorithm_type]:
                sa_algorithm_type_algorithm_times[r.sa_algorithm_type][r.algorithm] = [r.time_seconds]
            else:
                sa_algorithm_type_algorithm_times[r.sa_algorithm_type][r.algorithm].append(r.time_seconds)
    ###########



    ########### Writine Mean and Median CSVs 
    # for each we compute average and median
    algos: list[str] = []
    # all together
    analytics_column: list[AnalyticsColumn] = []
    # by macrogroups
    manual_macro_group_sa_algo_analytics_column: dict[str, dict[str, list[AnalyticsColumn]]] = {}
    # by representatives
    manual_representatives_macro_group_sa_algo_analytics_column: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    # by top
    manual_top_macro_group_sa_algo_analytics_column: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    # by manual custom macro groups
    manual_custom_macro_group_sa_algo_analytics_column: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    for sa_algo, algorithm_times in sa_algorithm_type_algorithm_times.items():
        algorithms_mean: dict[str, float] = {}
        algorithms_median: dict[str, float] = {}

        # which macrogroup?
        macro_group: str | None = None
        for mmg, sa_algos in MANUAL_MACRO_GROUPS.items():
            if sa_algo in sa_algos:
                macro_group = mmg
        
        if not macro_group:
            print(f"[WARNING]: Please consider inserting the algorithm: {sa_algo} in a macrogroup")

        # which custom macrogroup?
        custom_macro_groups: list[str] | None = None
        for cmmg, sa_algos in MANUAL_CUSTOM_MACRO_GROUPS.items():
            if sa_algo in sa_algos:
                if not custom_macro_groups:
                    custom_macro_groups = []
                custom_macro_groups.append(cmmg)
        # can be None, no warning
    
        for algo, times in algorithm_times.items():
            if algo not in algos:
                algos.append(algo)

            np_times = np.array(times)
            mean = np_times.mean()

            np_times.sort()
            ntimes = len(np_times)
            median = np_times[ntimes // 2] if ntimes % 2 != 0 else (np_times[ntimes // 2 - 1] + np_times[ntimes // 2]) / 2

            algorithms_mean[algo] = mean
            algorithms_median[algo] = median

        # all analytics
        ac = AnalyticsColumn(sa_algo, algorithms_mean, algorithms_median)
        analytics_column.append(ac)

        # macro groups analytics
        if macro_group:
            append_nested(
                manual_macro_group_sa_algo_analytics_column,
                macro_group,
                sa_algo,
                ac
            )

        # representatives
        if macro_group and macro_group in MANUAL_REPRESENTATIVES and sa_algo in MANUAL_REPRESENTATIVES[macro_group]:
            arc = ac.__to_representative__(macro_group)
            append_nested(
                manual_representatives_macro_group_sa_algo_analytics_column,
                macro_group,
                sa_algo,
                arc
            )

        # top
        if macro_group and macro_group in MANUAL_TOP and sa_algo in MANUAL_TOP[macro_group]:
            arc = ac.__to_representative__(macro_group)
            append_nested(
                manual_top_macro_group_sa_algo_analytics_column,
                macro_group,
                sa_algo,
                arc
            )

        # manual custom top
        if custom_macro_groups:
            # custom_macro_group = algo_custom_macrogroup(algo, custom_macro_groups, MANUAL_CUSTOM_MACRO_GROUPS)

            # if custom_macro_group:
            for custom_macro_group in custom_macro_groups:
                arc = ac.__to_representative__(custom_macro_group)
                append_nested(
                    manual_custom_macro_group_sa_algo_analytics_column,
                    custom_macro_group,
                    sa_algo,
                    arc
                )

    # write all together
    # order by sa_name
    analytics_column.sort(key=lambda x: x.sa_algorithm_type)
    write_analytics_csv(BASE_PATH, BENCHMARKS, analytics_column)

    # write by macrogroup
    for mmg in manual_macro_group_sa_algo_analytics_column:
        mmg_acss: list[AnalyticsColumn] = []
        for sa_algo, acs in manual_macro_group_sa_algo_analytics_column[mmg].items():
            mmg_acss += acs
        
        # order by name
        mmg_acss.sort(key=lambda x: x.sa_algorithm_type)
        write_analytics_csv(BASE_PATH / mmg, BENCHMARKS, mmg_acss, id=f"{mmg}-manual")

    # write by custom macrogroup
    for cmmg in manual_custom_macro_group_sa_algo_analytics_column:
        cmmg_acss: list[AnalyticsRepresentativesColumn] = []
        for sa_algo, acs in manual_custom_macro_group_sa_algo_analytics_column[cmmg].items():
            cmmg_acss += acs
        
        # order by name
        cmmg_acss.sort(key=lambda x: x.sa_algorithm_type)
        write_analytics_csv(BASE_PATH / cmmg, BENCHMARKS, cmmg_acss, id=f"{cmmg}-custom")

    # write by representatives
    arcss: list[AnalyticsRepresentativesColumn] = []
    for mmg in manual_representatives_macro_group_sa_algo_analytics_column:
        for sa_algo, arcs in manual_representatives_macro_group_sa_algo_analytics_column[mmg].items():
            arcss += arcs
    arcss.sort(key=lambda x: x.sa_algorithm_type)
    write_analytics_csv(BASE_PATH / REPRESENTATIVES_PATH, BENCHMARKS, arcss, id=f"manual")

    # write by top
    top_arcss: list[AnalyticsRepresentativesColumn] = []
    for mmg in manual_top_macro_group_sa_algo_analytics_column:
        for sa_algo, top_arcs in manual_top_macro_group_sa_algo_analytics_column[mmg].items():
            top_arcss += top_arcs
    top_arcss.sort(key=lambda x: x.sa_algorithm_type)
    write_analytics_csv(BASE_PATH / TOP_PATH, BENCHMARKS, top_arcss, id=f"top-manual")


    # write computed top
    # compute weights
    WEIGHTS = {}
    for r in rows:
        if len(WEIGHTS) < len(BENCHMARKS):
            if r.algorithm not in WEIGHTS:
                WEIGHTS[r.algorithm] = r.dfg_nodes
        else:
            break
    computed_mean_top_macro_group_sa_algo_analytics_column = return_top_algorithms(sa_algorithm_type_algorithm_times, BENCHMARKS, WEIGHTS, MANUAL_MACRO_GROUPS)
    computed_median_top_macro_group_sa_algo_analytics_column = return_top_algorithms(sa_algorithm_type_algorithm_times, BENCHMARKS, WEIGHTS, MANUAL_MACRO_GROUPS)

    # by mean
    top_mean_computed_arcss: list[AnalyticsRepresentativesColumn] = []
    for mmg in computed_mean_top_macro_group_sa_algo_analytics_column:
        for sa_algo, top_arc in computed_mean_top_macro_group_sa_algo_analytics_column[mmg].items():
            top_mean_computed_arcss.append(top_arc)
    top_mean_computed_arcss.sort(key=lambda x: x.sa_algorithm_type)

    TOP_MEAN_COMPUTED = "top-mean-computed"
    write_analytics_csv(BASE_PATH / Path(TOP_MEAN_COMPUTED), BENCHMARKS, top_mean_computed_arcss, id=TOP_MEAN_COMPUTED)

    # by median
    top_median_computed_arcss: list[AnalyticsRepresentativesColumn] = []
    for mmg in computed_median_top_macro_group_sa_algo_analytics_column:
        for sa_algo, top_arc in computed_median_top_macro_group_sa_algo_analytics_column[mmg].items():
            top_median_computed_arcss.append(top_arc)
    top_median_computed_arcss.sort(key=lambda x: x.sa_algorithm_type)

    TOP_MEDIAN_COMPUTED = "top-median-computed"
    write_analytics_csv(BASE_PATH / Path(TOP_MEDIAN_COMPUTED), BENCHMARKS, top_median_computed_arcss, id=TOP_MEDIAN_COMPUTED)
    ###########



    ########### Plots Mean and Median
    # should be anticipated by write_analytics_csv with same path to create directory
    # by macrogroups
    for mmg in manual_macro_group_sa_algo_analytics_column:
        mmg_acss: list[AnalyticsColumn] = []
        for sa_algo, acs in manual_macro_group_sa_algo_analytics_column[mmg].items():
            mmg_acss += acs
        
        # manual macrogroup
        plot_mean_median_barcharts(BASE_PATH / mmg, mmg_acss)

    # by custom macrogroups
    for cmmg in manual_custom_macro_group_sa_algo_analytics_column:
        cmmg_acss: list[AnalyticsColumn] = []
        for sa_algo, acs in manual_custom_macro_group_sa_algo_analytics_column[cmmg].items():
            cmmg_acss += acs
        
        # custom macrogroup
        plot_mean_median_barcharts(BASE_PATH / cmmg, cmmg_acss)

    # by representatives
    plot_mean_median_barcharts(BASE_PATH / REPRESENTATIVES_PATH, arcss)

    # by top
    plot_mean_median_barcharts(BASE_PATH / TOP_PATH, top_arcss)

    # by computed top mean
    plot_mean_median_barcharts(BASE_PATH / TOP_MEAN_COMPUTED, top_mean_computed_arcss)
    # by computed top median
    plot_mean_median_barcharts(BASE_PATH / TOP_MEDIAN_COMPUTED, top_median_computed_arcss)
    ###########



    print("Analysis completed correctly")
