"""Built-in official benchmark adapters."""

from safejudge.datasets.adapters.mm_safetybench import MMSafetyBenchAdapter
from safejudge.datasets.adapters.mossbench import MOSSBenchAdapter
from safejudge.datasets.adapters.omni_safetybench import OmniSafetyBenchAdapter

__all__ = ["MMSafetyBenchAdapter", "MOSSBenchAdapter", "OmniSafetyBenchAdapter"]

