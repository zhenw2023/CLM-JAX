
import numpy as np

from clm_jax.forcing.ordered_mapping import (
    prepare_ordered_mapping, prepare_partitioned_mapping,
)


def prepare_glacier_mapping(case, mapping, mapping_type):
    tasks = int(case.value("NTASKS_CPL", "1"))
    strategy = case.value(mapping_type, "X").upper()
    if tasks <= 0:
        raise ValueError("glacier mapping requires a resolved positive NTASKS_CPL")
    if strategy != "Y" or tasks == 1:
        return prepare_ordered_mapping(mapping)
    owner = (np.arange(mapping.source_size, dtype=np.int64) * tasks
             // mapping.source_size)
    return prepare_partitioned_mapping(mapping, owner)
