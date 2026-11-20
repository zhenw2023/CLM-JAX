
from collections import defaultdict
from dataclasses import replace
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.compilation import jit_discrete_parameters, materialize_outputs
from clm_jax.ecosystem_execution import _RuntimeWoodProductsResult
from clm_jax.physics.cn_flux_layout import with_fieldwise_cn_fluxes
from clm_jax.physics.cn_state_update2 import CnStateUpdate2State
from clm_jax.physics.cn_wood_products import cn_wood_products
from clm_jax.physics.cn_zero import (
    _FIELD_LOCATIONS,
    get_cn_flux_field,
    set_cn_flux_field,
    soil_only_reset_scope,
)
from clm_jax.physics.ecosystem_dynamics import EcosystemDynamicsPreDrainagePrefixInputs
from clm_jax.physics.ecosystem_dynamics_carry import CropAccumulatorCarry
from clm_jax.physics.selected_rows_gpu import restore_selected_rows

_PLANS = {}


def _prefix_inputs(inputs):
    while not isinstance(inputs, EcosystemDynamicsPreDrainagePrefixInputs):
        inputs = getattr(inputs, inputs._fields[0])
    return inputs


def soil_filter_plan(inputs):
    prefix = _prefix_inputs(inputs)
    fire = inputs.through_state_update3_precision.through_fire_flux.through_fire_area
    topology = fire.fire_area.topology
    gridcells = topology.gridcell_latitude_degrees.size
    sources = (prefix.cn_zero.soil_column, prefix.cn_zero.soil_patch,
               topology.patch_column_index, topology.soil_column, topology.soil_patch)
    key = (*map(id, sources), gridcells)
    cacheable = all(isinstance(value, jax.Array) for value in sources)
    entry = _PLANS.get(key) if cacheable else None
    if entry is not None and all(
        source() is value and not value.is_deleted()
        for source, value in zip(entry[0], sources, strict=True)
    ):
        return entry[1]
    column, patch, patch_column, fire_column, fire_patch = map(np.asarray, sources)
    nc, npatch = column.size, patch.size
    plan = None
    if (min(nc, npatch) > 256 and len({nc, npatch, gridcells}) == 3
            and np.all((patch_column >= 0) & (patch_column < nc))
            and np.array_equal(column[patch_column], patch)
            and np.array_equal(column, fire_column)
            and np.array_equal(patch, fire_patch)):
        ci, pi = (np.flatnonzero(mask).astype(np.int32) for mask in (column, patch))
        if ci.size and pi.size and (ci.size < nc or pi.size < npatch):
            cm, pm = np.full(nc, -1, np.int32), np.full(npatch, -1, np.int32)
            cm[ci], pm[pi] = np.arange(ci.size), np.arange(pi.size)
            plan = tuple(map(jnp.asarray, (ci, pi, cm, pm)))
    if cacheable:
        def forget(reference):
            current = _PLANS.get(key)
            if current is not None and any(item is reference for item in current[0]):
                _PLANS.pop(key, None)

        _PLANS[key] = (tuple(ref(value, forget) for value in sources), plan)
    return plan


def _select(tree, plan, sources):
    ci, pi, cm, pm = plan
    nc, npatch, memo = cm.size, pm.size, {}

    def take(path, value):
        if not hasattr(value, "shape") or not value.ndim or value.shape[0] not in (nc, npatch):
            return value
        name = getattr(path[-1], "name", "")
        remap = name if name in {"column_patch_indices", "patch_column_index"} else ""
        key = id(value), remap
        if key not in memo:
            selected = value[ci if value.shape[0] == nc else pi]
            if remap == "column_patch_indices":
                selected = jnp.where(selected >= 0, pm[jnp.maximum(selected, 0)], -1)
            elif remap == "patch_column_index":
                selected = cm[selected]
            memo[key] = selected
            if not remap:
                sources[id(selected)] = selected, value
        return memo[key]

    return jax.tree.map_with_path(take, tree)


def _prefix_state(state):
    return (state.state_update3_precision.fire_flux.fire_area.wood_products
            .state_update2h_precision.state_update2.gap_mortality.litter_vertical_transport
            .soil_n_state_update.second_guardrail.first_guardrail.ecosystem
            .pre_decomposition.prefix)


def _full_products(inputs, initial, plan):
    products = (inputs.through_state_update3_precision.through_fire_flux
                .through_fire_area.through_wood_products.wood_products)
    original_fluxes = _prefix_state(initial).cn_fluxes
    pi = plan[1]

    def execute(_, current):
        compact_fluxes = (current.state_update2h_precision.state_update2.gap_mortality
                          .litter_vertical_transport.soil_n_state_update.second_guardrail
                          .first_guardrail.ecosystem.pre_decomposition.prefix.cn_fluxes)
        fluxes = original_fluxes
        for name in ("vegetation_carbon.wood_harvestc_patch",
                     "vegetation_carbon.grainc_to_cropprodc_patch",
                     "vegetation_nitrogen.wood_harvestn_patch",
                     "vegetation_nitrogen.grainn_to_cropprodn_patch"):
            value = get_cn_flux_field(original_fluxes, name).at[pi].set(
                get_cn_flux_field(compact_fluxes, name),
                indices_are_sorted=True, unique_indices=True,
            )
            fluxes = set_cn_flux_field(fluxes, name, value)
        result = cn_wood_products(
            products.wood_products, current.wood_products, fluxes, products.parameters,
        )
        return _RuntimeWoodProductsResult(
            result.state, result.diagnostics.carbon.total_product_loss_gridcell,
        )

    return materialize_outputs(execute)


def _restore(inputs, initial, output, plan, sources):
    ci, pi, cm, pm = plan
    nc, npatch = cm.size, pm.size
    memo, copies, defaults = {}, defaultdict(list), {}
    area = initial.state_update3_precision.fire_flux.fire_area
    transport = (area.wood_products.state_update2h_precision.state_update2
                 .gap_mortality.litter_vertical_transport)
    soil = transport.soil_n_state_update
    guarded = soil.second_guardrail.first_guardrail
    prefix = guarded.ecosystem.pre_decomposition.prefix
    prefix_inputs = _prefix_inputs(inputs)
    deposition = prefix_inputs.nitrogen_deposition
    ndep = deposition.forcing_ndep_gridcell[deposition.column_gridcell_index]
    ndep_group, ndep_slot = _FIELD_LOCATIONS["soil_nitrogen.ndep_to_sminn_col"]
    potential = prefix_inputs.soil_biogeochem_potential_parameters
    annual_defaults = {
        "temporary_average_air_temperature_patch": (
            prefix.phenology_climate.temperature_average_2m_patch),
        "temporary_sum_potential_gpp_patch": (
            prefix.plant_nutrient_demand.temporary_sum_potential_gpp_patch),
        "temporary_maximum_retranslocated_nitrogen_patch": (
            prefix.plant_nutrient_demand.temporary_maximum_retranslocated_nitrogen_patch),
    }

    def scatter(path, previous, current):
        if not hasattr(current, "shape") or not current.ndim:
            return current
        if id(current) in memo:
            return memo[id(current)]
        if id(current) in sources:
            return sources[id(current)][1]
        field = getattr(path[-1], "name", "")
        if jax.tree_util.keystr(path).endswith(f".cn_fluxes.{ndep_group}.values[{ndep_slot}]"):
            previous = ndep
        elif field == "plant_nitrogen_demand_column":
            if field not in defaults:
                defaults[field] = jnp.zeros((nc,), current.dtype)
            previous = defaults[field]
        elif field in {"potential_carbon_loss_column_level_transition",
                       "potential_mineral_nitrogen_flux_column_level_transition"}:
            key = id(previous)
            if key not in defaults:
                production = (
                    (jnp.arange(previous.shape[1])[:, None] < potential.decomposition_level_count)
                    & (jnp.arange(previous.shape[2])[None, :]
                       < potential.decomposition_transition_count)
                    & ~jnp.asarray(potential.use_fates)
                )
                defaults[key] = jnp.where(production[None, :, :], 0., previous)
            previous = defaults[key]
        elif (len(path) == 2 and getattr(path[0], "name", "") == "annual"
              and field in annual_defaults):
            previous = annual_defaults[field]
        if ((previous.shape[0] == nc and current.shape[0] == ci.size)
                or (previous.shape[0] == npatch and current.shape[0] == pi.size)):
            placeholder = object()
            copies[(previous.shape, current.dtype)].append((placeholder, current, previous))
            memo[id(current)] = placeholder
        else:
            memo[id(current)] = current
        return memo[id(current)]

    full_state = jax.tree.map_with_path(scatter, initial, output.ecosystem.state)
    crop = prefix.crop_phenology
    carry_initial = output.post_drainage_carry._replace(
        carbon_nitrogen_state=CnStateUpdate2State(
            guarded.vegetation_carbon_nitrogen,
            transport.vertical_transport.carbon_pool_column_level_pool,
            transport.vertical_transport.nitrogen_pool_column_level_pool),
        mineral_nitrogen_state=soil.soil_nitrogen,
        cn_fluxes=prefix.cn_fluxes,
        total_burned_area_fraction_column=area.fire_area.total_burned_area_fraction_column,
        harvest_date_patch=crop.harvest_date_patch,
        crop_accumulators=CropAccumulatorCarry(
            crop.crop_live_patch, crop.growing_degree_days_since_planting_patch,
            crop.vernalization_factor_patch),
        excess_maintenance_respiration_carbon_loss_patch=soil.second_guardrail.xsmrpool_loss_patch,
        oxygen_decomposition_scalar_column_level=prefix.decomposition_scalars.oxygen_scalar_column_level,
        annual_sum_npp_patch=initial.annual.annual_sum_npp_patch,
        temporary_sum_npp_patch=initial.annual.temporary_sum_npp_patch,
    )
    result = output._replace(
        ecosystem=output.ecosystem._replace(state=full_state),
        post_drainage_carry=jax.tree.map_with_path(
            scatter, carry_initial, output.post_drainage_carry),
    )
    restored = {}
    for (shape, _), group in copies.items():
        inverse = cm if shape[0] == nc else pm
        for start in range(0, len(group), 8):
            selected = group[start:start + 8]
            values = restore_selected_rows(
                tuple(x[1] for x in selected), tuple(x[2] for x in selected), inverse,
            )
            restored.update((id(x[0]), value) for x, value in zip(selected, values, strict=True))
    return jax.tree.map(lambda value: restored.get(id(value), value), result)


def compact_executor(run, stages, *, share_inputs=False):
    def complete(inputs, state, plan):
        sources = {}
        selected = _select((inputs, state), plan, sources)
        compact_stages = replace(stages, wood_products=_full_products(inputs, state, plan))
        with soil_only_reset_scope():
            output = run(*selected, compact_stages)
        return _restore(inputs, state, output, plan, sources)

    return jit_discrete_parameters(
        with_fieldwise_cn_fluxes(complete, pack_output=False),
        forward_inputs=True, share_inputs=share_inputs,
    )
