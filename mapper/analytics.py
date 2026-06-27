from datetime import datetime
import json
from pathlib import Path
import numpy as np

from analytics.csv import AnalyticsColumn, load_all_rows, write_analytics_csv
from analytics.models import AnalyticsRepresentativesColumn, Row
from analytics.services import return_top_algorithms
from analytics.plots import plot_mean_median_barcharts

# analytics definitely needs a refactor

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

        # cooling
        "COOLING_RANDOM-NODE",
        "RANDOM-START_COOLING_RANDOM-NODE",
        "COOLING_RANDOM-NODE-WITH-SWAP",

        # morpher and sma
        "TEMPERATURE-SMA_RANDOM-NODE",
        "MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE",
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_RANDOM-NODE",

        # morpher and sma with swap
        "MORPHER-SMA-BEST-AND-MAX-CAP_RANDOM-NODE-WITH-SWAP",
    ],

    "list-selection": [
        # new name for latest benchmarks
        "RANDOM-START_COOLING-RESET_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # poisson distribution fixed
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # poisson proportional to operation nodes array
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-PROPORTIONAL-0.25",

        # poisson worst positioned
        "MORPHEUS-SMA-BEST-AND-MAX-CAP_WORST-POSITIONED-NODE_POISSON-PROPORTIONAL-TEMPERATURE",


        "RANDOM-START_COOLING_WORST-POSITIONED-NODE_POISSON-FIXED-2",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2",
    ],

    "spike-temperature": [
        # t reset
        "MORPHER-RESET-SMA_RANDOM-NODE",

        # t and sma reset
        "MORPHER-RESET-SMA-RESET-SMA_RANDOM-NODE",
        "MORPHER-RESET-SMA-RESET-SMA_RANDOM-NODE-WITH-SWAP",

        # random node
        "COOLING-RESET_RANDOM-NODE",
        "RANDOM-START_COOLING-RESET_RANDOM-NODE",

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
        # "COOLING_RANDOM-NODE",

        # because of new chnages random-start is added to the id, but remains classic implementation
        "RANDOM-START_COOLING_RANDOM-NODE",
    ],

    "list-selection": [
        # TODO USE BOTH FOR barplot

        # it is how we would have tested at the time
        # this shows we are stuck
        # did around 20s
        # used as implementation representative
        "RANDOM-START_COOLING_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # comparison of random one with reheat, as was first implemented and benchmarked with wrong version
        # around 60s
        # "RANDOM-START_COOLING-RESET_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # benchmark done, good
        # this is definitely the best
        # around 120s
        # "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2",

        # used as best algo
        # compare greedy with random start
        # around 90s
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2",
    ],

    "spike-temperature": [
        # "COOLING-RESET_RANDOM-NODE",

        # because of new chnages random-start is added to the id, but remains classic implementation
        "RANDOM-START_COOLING-RESET_RANDOM-NODE",
    ],

    "probability-reheating-routine": [
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],

    "precomputed-start": [
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP",
    ],
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

    "true-list-selection": [
        "RANDOM-START_COOLING_WORST-POSITIONED-NODE_POISSON-FIXED-2",
        "RANDOM-START_COOLING-RESET_WORST-POSITIONED-NODE_POISSON-FIXED-2",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2"
    ],

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
    ],

    "T-1-diff-init-conf-model-num": [
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
    ],

    "T-10-diff-init-conf-model-num": [
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
    ],

    "T-100-diff-init-conf-model-num": [
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
    ]
}


def print_missing_or_timeouts_benchmarks(manual_macro_group_sa_algo_analytics_column: dict[str, dict[str, list[AnalyticsColumn]]]):
    # print to check timeouts
    for mg, sa_algo_analytics_column in manual_macro_group_sa_algo_analytics_column.items():
        for sa_algo, analytics_columns in sa_algo_analytics_column.items():
            ac: AnalyticsColumn = analytics_columns[0]

            mean_missing: list[str] = []
            for benchmark in ac.algorithm_mean_time:
                if not ac.algorithm_mean_time[benchmark]:
                    mean_missing.append(benchmark)
            median_missing: list[str] = []
            for benchmark in ac.algorithm_mean_time:
                if not ac.algorithm_mean_time[benchmark]:
                    median_missing.append(benchmark)

            mean_benchmark_time_timeout: list[tuple[str, float]] = []
            for benchmark in ac.algorithm_mean_time:
                if ac.algorithm_mean_timeout[benchmark]:
                    mean_benchmark_time_timeout.append((benchmark, ac.algorithm_mean_time[benchmark]))

            median_benchmark_time_timeout: list[tuple[str, float]] = []
            for benchmark in ac.algorithm_median_time:
                if ac.algorithm_median_timeout[benchmark]:
                    median_benchmark_time_timeout.append((benchmark, ac.algorithm_median_time[benchmark]))

            if 0 < len(mean_missing) or 0 < len(median_missing) or 0 < len(mean_benchmark_time_timeout) or 0 < len(median_benchmark_time_timeout):
                print()
                print(mg)
                print(sa_algo)
                
                for memiss in mean_missing:
                    print(f"    mean missing - benchmark: {memiss}")
                for mdmiss in median_missing:
                    print(f"    median missing - benchmark: {mdmiss}")
                for mean_tou in mean_benchmark_time_timeout:
                    print(f"    mean TO - benchmark: {mean_tou[0]} - time: {mean_tou[1]}")
                for mean_tou in median_benchmark_time_timeout:
                    print(f"    median TO - benchmark: {mean_tou[0]} - time: {mean_tou[1]}")


def compute_normalized_times_by_group(
    data: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]]
) -> None:
    maximum_values_mean: dict[str, int] = {}
    maximum_values_median: dict[str, int] = {}

    for mmg in data:
        for _, arcs in data[mmg].items():
            c: AnalyticsRepresentativesColumn
            if isinstance(arcs, list):
                c = arcs[0]
            else:
                c = arcs

            for b, t in c.algorithm_mean_time.items():
                if not t:
                    continue
                if b not in maximum_values_mean or not maximum_values_mean[b] or maximum_values_mean[b] < t:
                    maximum_values_mean[b] = t

            for b, t in c.algorithm_median_time.items():
                if not t:
                    continue
                if b not in maximum_values_median or not maximum_values_median[b] or maximum_values_median[b] < t:
                    maximum_values_median[b] = t


    for mmg in data:
        for _, arcs in data[mmg].items():
            c: AnalyticsRepresentativesColumn
            if isinstance(arcs, list):
                c = arcs[0]
            else:
                c = arcs

            for b, t in c.algorithm_mean_time.items():
                c.algorithm_mean_time_norm_by_group[b] = t / maximum_values_mean[b] if t and maximum_values_mean[b] else None

            for b, t in c.algorithm_median_time.items():
                c.algorithm_median_time_norm_by_group[b] = t / maximum_values_median[b] if t and maximum_values_median[b] else None



def compute_normalized_times_by_strategy(
    data: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]],
    strategy: str
) -> None:
    strategy_values_mean: dict[str, int] = {}
    strategy_values_median: dict[str, int] = {}

    mmg_strategy: str

    print("\n\ntimes at entrance")
    for mmg in data:
        if strategy in data[mmg]:
            mmg_strategy = mmg
            arcs = data[mmg][strategy]

            c: AnalyticsRepresentativesColumn
            if isinstance(arcs, list):
                c = arcs[0]
            else:
                c = arcs

            strategy_values_mean = c.algorithm_mean_time.copy()
            strategy_values_median = c.algorithm_median_time.copy()

            for b, t in c.algorithm_mean_time.items():
                if c.algorithm_mean_timeout[b]:
                    strategy_values_mean[b] = None
                if c.algorithm_median_timeout[b]:
                    strategy_values_median[b] = None

    print(json.dumps(strategy_values_median, sort_keys=True, indent=4))

    # collect empty benchmarks
    empty_benchmarks_mean = []
    empty_benchmarks_median = []

    for b, t in strategy_values_mean.items():
        if not t or t >= 3999.9:
            empty_benchmarks_mean.append(b)
            data[mmg_strategy][strategy][0].algorithm_mean_time[b] = None
    for b, t in strategy_values_median.items():
        if not t or t >= 3999.9:
            empty_benchmarks_median.append(b)
            data[mmg_strategy][strategy][0].algorithm_median_time[b] = None

    print("\n\nmissing median")
    print(json.dumps(empty_benchmarks_median, sort_keys=True, indent=4))

    empty_benchmarks_times_mean = {b: [] for b in empty_benchmarks_mean}
    empty_benchmarks_times_median = {b: [] for b in empty_benchmarks_median}
    for mmg in data:
        for sa_algo, arcs in data[mmg].items():

            c: AnalyticsRepresentativesColumn
            if isinstance(arcs, list):
                c = arcs[0]
            else:
                c = arcs

            print(json.dumps(c.algorithm_mean_time, sort_keys=True, indent=4))
            print(json.dumps(c.algorithm_median_time, sort_keys=True, indent=4))

            # collect mean
            for b in empty_benchmarks_mean:
                if c.algorithm_mean_time[b]:
                    if c.algorithm_mean_timeout[b] or 3999.9 <= c.algorithm_mean_time[b]:
                        c.algorithm_mean_time[b] = None
                    else:
                        empty_benchmarks_times_mean[b].append(c.algorithm_mean_time[b])
            # collect median
            for b in empty_benchmarks_median:
                print(f"we can collect time {c.algorithm_median_time[b]} from {sa_algo} for {b}")
                if c.algorithm_median_time[b]:
                    if c.algorithm_median_timeout[b] or 3999.9 <= c.algorithm_median_time[b]:
                        c.algorithm_median_time[b] = None
                    else:
                        empty_benchmarks_times_median[b].append(c.algorithm_median_time[b])

    print(json.dumps(empty_benchmarks_times_median, sort_keys=True, indent=4))

    # sort arrays
    for b in empty_benchmarks_mean:
        empty_benchmarks_times_mean[b] = sorted(empty_benchmarks_times_mean[b])
    for b in empty_benchmarks_median:
        empty_benchmarks_times_median[b] = sorted(empty_benchmarks_times_median[b])

    # apply slowest times
    for b in empty_benchmarks_mean:
        strategy_values_mean[b] = empty_benchmarks_times_mean[b][-1] if len(empty_benchmarks_times_mean[b]) > 0 else None
    for b in empty_benchmarks_median:
        strategy_values_median[b] = empty_benchmarks_times_median[b][-1] if len(empty_benchmarks_times_median[b]) > 0 else None


    if len(strategy_values_mean) == 0 or len(strategy_values_median) == 0:
        AssertionError(f"Could not find strategy: {strategy}")
    
    print("\n\nmedian used for normalization")
    print(json.dumps(strategy_values_median, sort_keys=True, indent=4))

    for mmg in data:
        for sa_algo, arcs in data[mmg].items():
            c: AnalyticsRepresentativesColumn
            if isinstance(arcs, list):
                c = arcs[0]
            else:
                c = arcs
            print(sa_algo)
            print(json.dumps(c.algorithm_median_time, sort_keys=True, indent=4))
            print(json.dumps(c.algorithm_median_timeout, sort_keys=True, indent=4))

            for b, t in c.algorithm_mean_time.items():
                c.algorithm_mean_time_norm_by_group[b] = t / strategy_values_mean[b] if t and strategy_values_mean[b] else None

            for b, t in c.algorithm_median_time.items():
                c.algorithm_median_time_norm_by_group[b] = t / strategy_values_median[b] if t and strategy_values_median[b] else None



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

    # check for duplicate: naive
    for i in range(len(algos)):
        for j in range(len(algos)):
            if i != j and algos[i] == algos[j]:
                return True
    return False



if __name__ == "__main__":
    ########### Configurations Assertions 
    # ensure uniqueness
    for k, v in ALL_GROUPS_ALGO_UNIQUES.items():
        if not are_strategies_unique(v):
            print(f"Unique algorithms group doesn't have unique algorithms: {k}")
            exit(1)
    ###########



    ########### read all CSV files from /benchmark-results
    raw_rows: list[Row] = load_all_rows(CSV_BASE_PATH, CSV_NAME)
    print(f"We have: {len(raw_rows)} rows")
    ###########



    ########### row filter by date
    no_cutoff = False

    before_cutoff = False
    # could be set as CLI arg
    cutoff_date = datetime.strptime("21.05.26", "%d.%m.%y")

    # id
    # "" means no cutoff date
    id = ""

    if not no_cutoff:
        # is theres a cutoff date
        date: str = cutoff_date.isoformat().split("T")[0]

        if before_cutoff:
            # before
            id = f"before-{date}"

            before_rows = len(raw_rows)
            raw_rows = [r for r in raw_rows if r.id < cutoff_date]

            print(f"We had {before_rows} rows and before {date} we have {len(raw_rows)} rows")
        else:
            # after
            id = f"after-{date}"

            before_rows = len(raw_rows)
            raw_rows = [r for r in raw_rows if r.id > cutoff_date]
        
            print(f"We had {before_rows} rows and after {date} we have {len(raw_rows)} rows")

        BASE_PATH = BASE_PATH / id

    # make dir preentively
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    ###########




    ########### Collecting by SA_ALGO all its ALGORITHM BENCHMARKS all their 2, 5, 10, 20 TIMES
    sa_algorithm_type_algorithm_rows_2x2: dict[str, dict[str, list[Row]]] = {}
    sa_algorithm_type_algorithm_rows_5x5: dict[str, dict[str, list[Row]]] = {}
    sa_algorithm_type_algorithm_rows_10x10: dict[str, dict[str, list[Row]]] = {}
    sa_algorithm_type_algorithm_rows: dict[str, dict[str, list[Row]]] = {}
    for r in raw_rows:
        if r.size_x != r.size_y:
            print(f"Skipping {r.size_x}x{r.size_y} mixed size")
            continue

        match r.size_x:
            case 2:
                append_nested(sa_algorithm_type_algorithm_rows_2x2, r.sa_algorithm_type, r.algorithm, r)
            case 5:
                append_nested(sa_algorithm_type_algorithm_rows_5x5, r.sa_algorithm_type, r.algorithm, r)
            case 10:
                append_nested(sa_algorithm_type_algorithm_rows_10x10, r.sa_algorithm_type, r.algorithm, r)
            case 20:
                append_nested(sa_algorithm_type_algorithm_rows, r.sa_algorithm_type, r.algorithm, r)
            case _:
                print(f"Size {r.size_x}x{r.size_y} not handled")

        # if r.sa_algorithm_type not in sa_algorithm_type_algorithm_rows:
        #     sa_algorithm_type_algorithm_rows[r.sa_algorithm_type] = {}
        #     sa_algorithm_type_algorithm_rows[r.sa_algorithm_type][r.algorithm] = [r]
        # else:
        #     if r.algorithm not in sa_algorithm_type_algorithm_rows[r.sa_algorithm_type]:
        #         sa_algorithm_type_algorithm_rows[r.sa_algorithm_type][r.algorithm] = [r]
        #     else:
        #         sa_algorithm_type_algorithm_rows[r.sa_algorithm_type][r.algorithm].append(r)
    ###########


    ####### Compute Means and Medians for Manual representatives for 2, 5, 10 sizes
    manual_representatives_macro_group_sa_algo_analytics_column_2x2: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    manual_representatives_macro_group_sa_algo_analytics_column_5x5: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    manual_representatives_macro_group_sa_algo_analytics_column_10x10: dict[str, dict[str, list[AnalyticsRepresentativesColumn]]] = {}
    for size_smaller, (rows_smaller, dicts_smaller) in zip(
        [2, 5, 10],
        zip(
            [
                sa_algorithm_type_algorithm_rows_2x2,
                sa_algorithm_type_algorithm_rows_5x5,
                sa_algorithm_type_algorithm_rows_10x10
            ],
            [
                manual_representatives_macro_group_sa_algo_analytics_column_2x2,
                manual_representatives_macro_group_sa_algo_analytics_column_5x5,
                manual_representatives_macro_group_sa_algo_analytics_column_10x10
            ]
        )
    ):
        print(f"\n\nFor size {size_smaller}")
        if len(rows_smaller) == 0:
            print("     no rows")

        for sa_algo, algorithm_rows in rows_smaller.items():
            algorithms_mean_timeout: dict[str, bool | None] = {}
            algorithms_median_timeout: dict[str, bool | None] = {}

            algorithms_mean: dict[str, float | None] = {}
            algorithms_median: dict[str, float | None] = {}

            # this norm is only by within the single algorithm times for each benchmarks
            algorithms_mean_norm: dict[str, float | None] = {}
            algorithms_median_norm: dict[str, float | None] = {}

            algorithms_mean_iteration: dict[str, float | None] = {}
            algorithms_median_iteration: dict[str, float | None] = {}

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

            is_representative = macro_group and macro_group in MANUAL_REPRESENTATIVES and sa_algo in MANUAL_REPRESENTATIVES[macro_group]
            if is_representative:
                # if representative how many benchmarks do we have per benchmarks:
                print(f"{macro_group} - {sa_algo}")

            alogs_and_rows: dict[str, int] = {}
            for algo, rows in algorithm_rows.items():
                if is_representative:
                    alogs_and_rows[algo] = rows

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

                algorithms_mean_norm[algo] = mean / np_times.max() if np_times.max() > 0 else 0
                algorithms_median_norm[algo] = median  / np_times.max() if np_times.max() > 0 else 0

                algorithms_mean_iteration[algo] = mean_iterations / np_iterations.max()
                algorithms_median_iteration[algo] = median_iterations / np_iterations.max()

            if is_representative:
                # TODO we need to get 20 benchmarks run for each representative in each benchmark
                # used to check quantities
                for b in BENCHMARKS:
                    # if representative how many benchmarks do we have per benchmarks:
                    if b in alogs_and_rows:
                        print(f"    {b}: {len(alogs_and_rows[b])}")
                    else:
                        print(f"    {b}: 0")

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

            # representatives
            if is_representative:
                arc = AnalyticsRepresentativesColumn(
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
                )
                append_nested(
                    dicts_smaller,
                    macro_group,
                    sa_algo,
                    arc
                )

    # should only write CSVs for 2x2, 5x5, 10x10

    # write by representatives
    for mmg_size, mmg_dict in zip(
        [2, 5, 10],
        [
            manual_representatives_macro_group_sa_algo_analytics_column_2x2,
            manual_representatives_macro_group_sa_algo_analytics_column_5x5,
            manual_representatives_macro_group_sa_algo_analytics_column_10x10
        ]
    ):
        arcss_by_size: list[AnalyticsRepresentativesColumn] = []
        for mmg in mmg_dict:
            for sa_algo, arcs_by_size in mmg_dict[mmg].items():
                arcss_by_size += arcs_by_size
        arcss_by_size.sort(key=lambda x: x.sa_algorithm_type)
        write_analytics_csv(BASE_PATH / REPRESENTATIVES_PATH, BENCHMARKS, arcss_by_size, id=f"manual-{mmg_size}")



    # "detail": list[acolumns] is useless, will always be 1 element
    ########### Compute Mean and Median
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
    
    print("\n\nFor size 20")
    for sa_algo, algorithm_rows in sa_algorithm_type_algorithm_rows.items():
        algorithms_mean_timeout: dict[str, bool | None] = {}
        algorithms_median_timeout: dict[str, bool | None] = {}

        algorithms_mean: dict[str, float | None] = {}
        algorithms_median: dict[str, float | None] = {}

        # this norm is only by within the single algorithm times for each benchmarks
        algorithms_mean_norm: dict[str, float | None] = {}
        algorithms_median_norm: dict[str, float | None] = {}

        algorithms_mean_iteration: dict[str, float | None] = {}
        algorithms_median_iteration: dict[str, float | None] = {}

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

        is_representative = macro_group and macro_group in MANUAL_REPRESENTATIVES and sa_algo in MANUAL_REPRESENTATIVES[macro_group]
        if is_representative:
            # if representative how many benchmarks do we have per benchmarks:
            print(f"{macro_group} - {sa_algo}")

        alogs_and_rows: dict[str, int] = {}
        for algo, rows in algorithm_rows.items():
            if is_representative:
                alogs_and_rows[algo] = rows

            if algo not in algos:
                algos.append(algo)

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
            algorithms_median_norm[algo] = median  / np_times.max()

            algorithms_mean_iteration[algo] = mean_iterations / np_iterations.max()
            algorithms_median_iteration[algo] = median_iterations / np_iterations.max()

        if is_representative:
            # TODO we need to get 20 benchmarks run for each representative in each benchmark
            # used to check quantities
            for b in BENCHMARKS:
                # if representative how many benchmarks do we have per benchmarks:
                if b in alogs_and_rows:
                    print(f"    {b}: {len(alogs_and_rows[b])}")
                else:
                    print(f"    {b}: 0")

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

        # all analytics
        ac = AnalyticsColumn(
            sa_algo,
            algorithms_mean_timeout,
            algorithms_median_timeout,
            algorithms_mean,
            algorithms_median,
            algorithms_mean_norm,
            algorithms_median_norm,
            algorithms_mean_iteration,
            algorithms_median_iteration
        )
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
        if is_representative:
            arc = ac.__to_representative__(
                macro_group
            )
            append_nested(
                manual_representatives_macro_group_sa_algo_analytics_column,
                macro_group,
                sa_algo,
                arc
            )

        # top
        if macro_group and macro_group in MANUAL_TOP and sa_algo in MANUAL_TOP[macro_group]:
            arc = ac.__to_representative__(
                macro_group
            )
            append_nested(
                manual_top_macro_group_sa_algo_analytics_column,
                macro_group,
                sa_algo,
                arc
            )

        # manual custom top
        if custom_macro_groups:
            for custom_macro_group in custom_macro_groups:
                arc = ac.__to_representative__(
                    custom_macro_group
                )
                append_nested(
                    manual_custom_macro_group_sa_algo_analytics_column,
                    custom_macro_group,
                    sa_algo,
                    arc
                )










    # actually normalize by group for AnalyticsRepresentativesColumn
    compute_normalized_times_by_strategy(
        manual_representatives_macro_group_sa_algo_analytics_column,
        "RANDOM-START_COOLING_RANDOM-NODE")
    # print(manual_representatives_macro_group_sa_algo_analytics_column)
    compute_normalized_times_by_group(
        manual_top_macro_group_sa_algo_analytics_column)
    compute_normalized_times_by_group(
        manual_custom_macro_group_sa_algo_analytics_column)


















    ########### Save CSVs
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
    for r in raw_rows:
        if len(WEIGHTS) < len(BENCHMARKS):
            if r.algorithm not in WEIGHTS:
                WEIGHTS[r.algorithm] = r.dfg_nodes
        else:
            break

    computed_mean_top_macro_group_sa_algo_analytics_column = return_top_algorithms(sa_algorithm_type_algorithm_rows, BENCHMARKS, WEIGHTS, MANUAL_MACRO_GROUPS)
    computed_median_top_macro_group_sa_algo_analytics_column = return_top_algorithms(sa_algorithm_type_algorithm_rows, BENCHMARKS, WEIGHTS, MANUAL_MACRO_GROUPS)

    compute_normalized_times_by_group(
        computed_mean_top_macro_group_sa_algo_analytics_column)
    compute_normalized_times_by_group(
        computed_median_top_macro_group_sa_algo_analytics_column)

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
        plot_mean_median_barcharts(
            BASE_PATH / mmg,
            mmg_acss,
            get_values=lambda row, attr: (
                row.algorithm_mean_time if attr == "mean" else row.algorithm_median_time
            ),
            data_type="")
        plot_mean_median_barcharts(
            BASE_PATH / mmg,
            mmg_acss,
            get_values=lambda row, attr: (
                row.algorithm_mean_iteration if attr == "mean" else row.algorithm_median_iteration
            ),
            data_type="")

    # by custom macrogroups
    for cmmg in manual_custom_macro_group_sa_algo_analytics_column:
        cmmg_acss: list[AnalyticsColumn] = []
        for sa_algo, acs in manual_custom_macro_group_sa_algo_analytics_column[cmmg].items():
            cmmg_acss += acs
        
        # custom macrogroup
        plot_mean_median_barcharts(
            BASE_PATH / cmmg,
            cmmg_acss,
            get_values=lambda row, attr: (
                row.algorithm_mean_time if attr == "mean" else row.algorithm_median_time
            ),
            data_type="")
        plot_mean_median_barcharts(
            BASE_PATH / cmmg,
            cmmg_acss,
            get_values=lambda row, attr: (
                row.algorithm_mean_iteration if attr == "mean" else row.algorithm_median_iteration
            ),
            data_type="")



    # by representatives, top, top mean and top median
    TO_PLOT_PATHS = [
        REPRESENTATIVES_PATH,
        TOP_PATH,
        TOP_MEAN_COMPUTED,
        TOP_MEDIAN_COMPUTED
    ]
    TO_PLOT_ITEMS = [
        arcss,
        top_arcss,
        top_mean_computed_arcss,
        top_median_computed_arcss
    ]

    algo_names_representatives = {
        "RANDOM-START_COOLING_RANDOM-NODE": "Base SA",
        "RANDOM-START_COOLING_WORST-POSITIONED-NODE_POISSON-FIXED-2": "Guided Selection SA",
        "RANDOM-START_COOLING-RESET_RANDOM-NODE": "Reheating SA",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP": "Guided Reheating SA",
        "GREEDY-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_RANDOM-NODE-WITH-SWAP": "Guided Starting State Generation and Reheating SA",
        "RANDOM-START_COOLING-RESET-TO-0-45-DYNAMIC-BEST-COST-START-T-COEFF-10_WORST-POSITIONED-NODE_POISSON-FIXED-2": "Guided Selection and Reheating SA",
    }

    for (path, items) in zip(TO_PLOT_PATHS, TO_PLOT_ITEMS):
        if len(items) == 0:
            continue

        print()
        print(f"printing for path: {path}")
            # by time
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time if attr == "mean" else row.algorithm_median_time
        #     ),
        #     data_type="Times",
        #     algo_names=algo_names_representatives)


                # time normalized both regular and inverse
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time_norm if attr == "mean" else row.algorithm_median_time_norm
        #     ),
        #     data_type="Times",
        #     algo_names=algo_names_representatives)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time_norm if attr == "mean" else row.algorithm_median_time_norm
        #     ),
        #     data_type="Times",
        #     opacity_by_norm=True,
        #     algo_names=algo_names_representatives)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time_norm if attr == "mean" else row.algorithm_median_time_norm
        #     ),
        #     data_type="Times",
        #     opacity_by_norm=True,
        #     inverse_normalized_opacity=True,
        #     algo_names=algo_names_representatives)


                # time normalized by group both regular and inverse
        plot_mean_median_barcharts(
            BASE_PATH / path,
            items,
            get_values=lambda row, attr: (
                row.algorithm_mean_time_norm_by_group if attr == "mean" else row.algorithm_median_time_norm_by_group
            ),
            data_type="Times",
            id="-by-group",
            algo_names=algo_names_representatives,
            LOG_SCALE=False)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time_norm_by_group if attr == "mean" else row.algorithm_median_time_norm_by_group
        #     ),
        #     data_type="Times",
        #     id="-by-group",
        #     opacity_by_norm=True,
        #     algo_names=algo_names_representatives)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_time_norm_by_group if attr == "mean" else row.algorithm_median_time_norm_by_group
        #     ),
        #     data_type="Times",
        #     opacity_by_norm=True,
        #     inverse_normalized_opacity=True,
        #     id="-by-group",
        #     algo_names=algo_names_representatives)


                # by iterations time both regular and inverse
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_iteration if attr == "mean" else row.algorithm_median_iteration
        #     ),
        #     data_type="Iterations",
        #     algo_names=algo_names_representatives)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_iteration if attr == "mean" else row.algorithm_median_iteration
        #     ),
        #     data_type="Iterations",
        #     opacity_by_norm=True,
        #     algo_names=algo_names_representatives)
        # plot_mean_median_barcharts(
        #     BASE_PATH / path,
        #     items,
        #     get_values=lambda row, attr: (
        #         row.algorithm_mean_iteration if attr == "mean" else row.algorithm_median_iteration
        #     ),
        #     data_type="Iterations",
        #     opacity_by_norm=True,
        #     inverse_normalized_opacity=True,
        #     algo_names=algo_names_representatives)
    ###########




    print("Analysis completed correctly")
