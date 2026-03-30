from procmem2skills.adapters.alfworld import PROFILE as ALFWORLD_PROFILE
from procmem2skills.adapters.mind2web import PROFILE as MIND2WEB_PROFILE
from procmem2skills.adapters.terminal_bench import PROFILE as TERMINAL_BENCH_PROFILE
from procmem2skills.adapters.webarena import PROFILE as WEB_ARENA_PROFILE

BENCHMARK_PROFILES = (
    WEB_ARENA_PROFILE,
    MIND2WEB_PROFILE,
    ALFWORLD_PROFILE,
    TERMINAL_BENCH_PROFILE,
)

__all__ = ["BENCHMARK_PROFILES"]
