
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.voc import VOCParameters

_MAPPED_SURFACE_VARIABLES = (
    "EF1_BTR",
    "EF1_FET",
    "EF1_FDT",
    "EF1_SHR",
    "EF1_GRS",
    "EF1_CRP",
)


@dataclass(frozen=True)
class VOCConfiguration:

    mechanism_names: tuple[str, ...]
    compound_names: tuple[str, ...]
    pft_names: tuple[str, ...]
    factor_file: Path
    parameters: VOCParameters

    @classmethod
    def from_case(cls, case: CimeCase) -> VOCConfiguration:
        group = read_case_namelist_group(case, "drv_flds_in", "megan_emis_nl")
        raw_specifier = group.get("megan_specifier", ())
        if isinstance(raw_specifier, str):
            specifiers = (raw_specifier,)
        else:
            specifiers = tuple(str(value) for value in raw_specifier if value is not None)
        mechanisms = _parse_specifiers(specifiers)
        factor_file = Path(case.expand(str(group["megan_factors_file"]))).expanduser().resolve()
        use_mapped = bool(group.get("megan_mapped_emisfctrs", False))
        return cls.from_factor_file(
            factor_file,
            mechanisms,
            use_mapped_isoprene_factors=use_mapped,
        )

    @classmethod
    def from_factor_file(
        cls,
        factor_file: str | Path,
        mechanisms: tuple[tuple[str, tuple[tuple[float, str], ...]], ...],
        *,
        use_mapped_isoprene_factors: bool = False,
    ) -> VOCConfiguration:
        factor_path = Path(factor_file).expanduser().resolve()
        if not factor_path.is_file():
            raise FileNotFoundError(f"MEGAN factor file does not exist: {factor_path}")
        if not mechanisms:
            raise ValueError("MEGAN configuration contains no mechanism compounds")

        mechanism_names = tuple(name for name, _ in mechanisms)
        if len(set(mechanism_names)) != len(mechanism_names):
            raise ValueError("MEGAN mechanism compound names must be unique")

        compound_names: list[str] = []
        compound_coefficients: list[float] = []
        compound_index: dict[str, int] = {}
        mechanism_terms: list[list[int]] = []
        for _, terms in mechanisms:
            if not terms:
                raise ValueError("every MEGAN mechanism compound requires at least one term")
            indices: list[int] = []
            for coefficient, compound_name in terms:
                if compound_name not in compound_index:
                    compound_index[compound_name] = len(compound_names)
                    compound_names.append(compound_name)
                    compound_coefficients.append(coefficient)
                indices.append(compound_index[compound_name])
            mechanism_terms.append(indices)

        with xr.open_dataset(factor_path, decode_cf=False, mask_and_scale=False) as dataset:
            table_names = _decode_character_rows(dataset["Comp_Name"].values)
            pft_names = _decode_character_rows(dataset["PFT_Name"].values)
            table_index = {name: index for index, name in enumerate(table_names)}
            missing = [name for name in compound_names if name not in table_index]
            if missing:
                raise ValueError(f"MEGAN compounds absent from factor file: {missing}")

            class_number_all = np.asarray(dataset["Class_Num"].values, dtype=np.int32)
            molecular_weight_all = np.asarray(dataset["Comp_MW"].values, dtype=np.float64)
            compound_ef = np.asarray(dataset["Comp_EF"].values, dtype=np.float64)
            class_ef = np.asarray(dataset["Class_EF"].values, dtype=np.float64)
            compound_classes = np.asarray(
                [class_number_all[table_index[name]] - 1 for name in compound_names],
                dtype=np.int32,
            )
            molecular_weights = np.asarray(
                [molecular_weight_all[table_index[name]] for name in compound_names],
                dtype=np.float64,
            )
            number_of_pfts = compound_ef.shape[0]
            factors = np.zeros((len(compound_names), number_of_pfts + 1), dtype=np.float64)
            for output_index, name in enumerate(compound_names):
                input_index = table_index[name]
                class_index = compound_classes[output_index]
                factors[output_index, 1:] = compound_ef[:, input_index] * class_ef[:, class_index]

            class_arrays = {
                name: np.asarray(dataset[name].values, dtype=np.float64)
                for name in (
                    "LDF",
                    "Agro",
                    "Amat",
                    "Anew",
                    "Aold",
                    "betaT",
                    "ct1",
                    "ct2",
                    "Ceo",
                )
            }

        maximum_terms = max(len(indices) for indices in mechanism_terms)
        mechanism_index = np.zeros((len(mechanism_terms), maximum_terms), dtype=np.int32)
        mechanism_mask = np.zeros((len(mechanism_terms), maximum_terms), dtype=np.bool_)
        for mechanism, indices in enumerate(mechanism_terms):
            mechanism_index[mechanism, : len(indices)] = indices
            mechanism_mask[mechanism, : len(indices)] = True

        non_evergreen, mapped_category = _pft_branch_tables(pft_names)
        parameters = VOCParameters(
            emission_factor_by_compound_pft=jnp.asarray(factors),
            compound_class=jnp.asarray(compound_classes),
            compound_molecular_weight=jnp.asarray(molecular_weights),
            compound_coefficient=jnp.asarray(compound_coefficients, dtype=jnp.float64),
            is_isoprene=jnp.asarray(
                [name == "isoprene" for name in compound_names], dtype=jnp.bool_
            ),
            mechanism_compound_index=jnp.asarray(mechanism_index),
            mechanism_compound_mask=jnp.asarray(mechanism_mask),
            light_dependent_fraction=jnp.asarray(class_arrays["LDF"]),
            growing_leaf_age_factor=jnp.asarray(class_arrays["Agro"]),
            mature_leaf_age_factor=jnp.asarray(class_arrays["Amat"]),
            new_leaf_age_factor=jnp.asarray(class_arrays["Anew"]),
            old_leaf_age_factor=jnp.asarray(class_arrays["Aold"]),
            temperature_beta=jnp.asarray(class_arrays["betaT"]),
            temperature_ct1=jnp.asarray(class_arrays["ct1"]),
            temperature_ct2=jnp.asarray(class_arrays["ct2"]),
            temperature_ceo=jnp.asarray(class_arrays["Ceo"]),
            non_evergreen_pft=jnp.asarray(non_evergreen),
            mapped_factor_category_by_pft=jnp.asarray(mapped_category),
            use_mapped_isoprene_factors=jnp.asarray(use_mapped_isoprene_factors),
        )
        return cls(
            mechanism_names=mechanism_names,
            compound_names=tuple(compound_names),
            pft_names=pft_names,
            factor_file=factor_path,
            parameters=parameters,
        )


def load_mapped_isoprene_emission_factors(case: CimeCase) -> jnp.ndarray:

    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    surface_path = Path(case.expand(str(group["fsurdat"]))).expanduser().resolve()
    with xr.open_dataset(surface_path, decode_cf=False, mask_and_scale=True) as dataset:
        fields = [
            np.asarray(dataset[name].load().values, dtype=np.float64).reshape(-1)
            for name in _MAPPED_SURFACE_VARIABLES
        ]
    return jnp.asarray(np.ascontiguousarray(np.stack(fields, axis=1)))


def load_mapped_isoprene_emission_factors_for_grid(
    case: CimeCase,
    topology: SubgridTopology,
) -> jnp.ndarray:

    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    surface_path = Path(case.expand(str(group["fsurdat"]))).expanduser().resolve()
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    grid_count = topology.grid.longitude_degrees.shape[0]
    if latitude.shape != (grid_count,) or longitude.shape != (grid_count,):
        raise ValueError("native surface-data indices must contain every land gridcell")
    with xr.open_dataset(surface_path, decode_cf=False, mask_and_scale=True) as dataset:
        selected: list[np.ndarray] = []
        source_shape: tuple[int, int] | None = None
        for name in _MAPPED_SURFACE_VARIABLES:
            source = np.asarray(dataset[name].load().values, dtype=np.float64)
            if source.ndim != 2:
                raise ValueError(f"surface-data {name} must have latitude/longitude axes")
            if source_shape is None:
                source_shape = source.shape
            elif source.shape != source_shape:
                raise ValueError("mapped isoprene surface fields do not share a grid")
            if (
                np.any(latitude < 0)
                or np.any(latitude >= source.shape[0])
                or np.any(longitude < 0)
                or np.any(longitude >= source.shape[1])
            ):
                raise ValueError("native land grid lies outside mapped isoprene surface fields")
            selected.append(source[latitude, longitude])
    values = np.ascontiguousarray(np.stack(selected, axis=1))
    if np.any(~np.isfinite(values)):
        raise ValueError("mapped isoprene factors contain non-finite land-grid values")
    return jnp.asarray(values, dtype=jnp.float64)


def _parse_specifiers(
    specifiers: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[float, str], ...]], ...]:
    mechanisms: list[tuple[str, tuple[tuple[float, str], ...]]] = []
    for expression in specifiers:
        if not expression.strip():
            continue
        if "=" not in expression:
            raise ValueError(f"MEGAN expression lacks '=': {expression!r}")
        mechanism, right = expression.split("=", 1)
        mechanism = mechanism.strip()
        if not mechanism:
            raise ValueError(f"MEGAN expression has an empty mechanism name: {expression!r}")
        terms: list[tuple[float, str]] = []
        for raw_term in right.split("+"):
            term = raw_term.strip()
            if not term:
                raise ValueError(f"MEGAN expression has an empty term: {expression!r}")
            if "*" in term:
                raw_coefficient, compound = term.split("*", 1)
                coefficient = float(raw_coefficient.strip().replace("d", "e").replace("D", "E"))
            else:
                coefficient = 1.0
                compound = term
            compound = compound.strip()
            if not compound:
                raise ValueError(f"MEGAN expression has an empty compound: {expression!r}")
            terms.append((coefficient, compound))
        mechanisms.append((mechanism, tuple(terms)))
    return tuple(mechanisms)


def _decode_character_rows(values: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype.kind not in "SU":
        raise ValueError("MEGAN name tables must be two-dimensional character arrays")
    rows = array
    decoded = tuple(
        b"".join(value if isinstance(value, bytes) else str(value).encode("utf-8") for value in row)
        .decode("utf-8")
        .rstrip("\x00 ")
        for row in rows
    )
    if any(not name for name in decoded):
        raise ValueError("MEGAN name table contains an empty entry")
    return decoded


def _pft_branch_tables(pft_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    expected = {
        "needleleaf_deciduous_boreal_tree": 3,
        "broadleaf_deciduous_tropical_tree": 6,
        "c3_crop": 15,
    }
    one_based = {name: index for index, name in enumerate(("not_vegetated", *pft_names))}
    for name, expected_index in expected.items():
        if one_based.get(name) != expected_index:
            raise ValueError(
                f"MEGAN PFT table has incompatible {name!r} index: "
                f"{one_based.get(name)!r}, expected {expected_index}"
            )

    number_of_types = len(pft_names) + 1
    non_evergreen = np.zeros(number_of_types, dtype=np.bool_)
    non_evergreen[3] = True
    non_evergreen[6:] = True

    mapped_category = np.full(number_of_types, -1, dtype=np.int32)
    mapped_category[1:3] = 1
    mapped_category[3] = 2
    mapped_category[4:9] = 0
    mapped_category[9:12] = 3
    mapped_category[12:15] = 4
    mapped_category[15:] = 5
    return non_evergreen, mapped_category
