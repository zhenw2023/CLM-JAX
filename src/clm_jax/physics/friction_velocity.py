
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, lax

from clm_jax.constants import GRAVITY, VON_KARMAN
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    fortran_product,
    fortran_rounding_barrier,
)
from clm_jax.physics.pointwise import scalar_map

MOMENTUM_TRANSITION = 1.574
SCALAR_TRANSITION = 0.465


class MoninObukhovInitialization(NamedTuple):
    stability_wind_speed: jnp.ndarray
    monin_obukhov_length: jnp.ndarray


class FrictionVelocityState(NamedTuple):

    momentum_stability_patch: jnp.ndarray
    deposition_velocity_patch: jnp.ndarray
    wind_10m_dust_patch: jnp.ndarray
    wind_10m_patch: jnp.ndarray
    atmospheric_wind_patch: jnp.ndarray
    dust_friction_velocity_patch: jnp.ndarray


class FrictionVelocityResult(NamedTuple):
    state: FrictionVelocityState
    friction_velocity_patch: jnp.ndarray
    temperature_profile_patch: jnp.ndarray
    humidity_profile_patch: jnp.ndarray
    temperature_profile_2m_patch: jnp.ndarray
    humidity_profile_2m_patch: jnp.ndarray
    momentum_numerator_patch: jnp.ndarray
    momentum_denominator_patch: jnp.ndarray
    momentum_zeta_patch: jnp.ndarray


def _validate_vector_shapes(size: int, values: tuple[tuple[str, jnp.ndarray], ...]) -> None:
    for name, value in values:
        if value.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")


def _strong_stable_denominator(
    stable_length: jnp.ndarray,
    roughness: jnp.ndarray,
    stable_zeta: jnp.ndarray,
) -> jnp.ndarray:
    final_group = fortran_add(
        fortran_add(
            fortran_product(jnp.float64(5.0), _elementwise_log(stable_zeta)),
            stable_zeta,
        ),
        jnp.full_like(stable_zeta, -1.0),
    )
    result = fortran_add(
        _elementwise_log(fortran_divide(stable_length, roughness)),
        jnp.full_like(stable_zeta, 5.0),
    )
    result = fortran_add(
        result,
        -fortran_divide(
            fortran_product(jnp.float64(5.0), roughness),
            stable_length,
        ),
    )
    return fortran_add(result, final_group)


def _elementwise_atan(value: jnp.ndarray) -> jnp.ndarray:

    if default_backend() == "gpu":
        from clm_jax.physics.native_arctangent import native_atan
        return native_atan(value)

    shape = value.shape
    flat = jnp.reshape(value, (-1,))
    return jnp.reshape(scalar_map(jnp.arctan, flat), shape)


def _elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    if default_backend() == "gpu":
        from clm_jax.physics.native_logarithm import native_log
        return native_log(value)
    shape = value.shape
    flat = jnp.reshape(value, (-1,))
    return jnp.reshape(scalar_map(jnp.log, flat), shape)


def _stability_power(value: jnp.ndarray, exponent: float) -> jnp.ndarray:

    if default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(value, jnp.float64(exponent))
    return value ** jnp.float64(exponent)


def _refined_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:

    return fortran_divide(numerator, denominator)


def _multiply_boundary(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(left, right)
    if default_backend() == "gpu":
        return fortran_multiply(left, right)
    factors = jnp.stack((right, jnp.ones_like(right)), axis=0)

    def multiply_step(running: jnp.ndarray, factor: jnp.ndarray):
        return running * factor, None

    result, _ = lax.scan(multiply_step, left, factors)
    return result


def stability_function_momentum(zeta: jnp.ndarray) -> jnp.ndarray:

    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    root_squared = jnp.sqrt(jnp.float64(1.0) - jnp.float64(16.0) * zeta)
    root = jnp.sqrt(root_squared)
    return (
        jnp.float64(2.0) * _elementwise_log((jnp.float64(1.0) + root) * jnp.float64(0.5))
        + _elementwise_log((jnp.float64(1.0) + root_squared) * jnp.float64(0.5))
        - jnp.float64(2.0) * _elementwise_atan(root)
        + jnp.float64(jnp.pi) * jnp.float64(0.5)
    )


def stability_function_scalar(zeta: jnp.ndarray) -> jnp.ndarray:

    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    root_squared = jnp.sqrt(jnp.float64(1.0) - jnp.float64(16.0) * zeta)
    return jnp.float64(2.0) * _elementwise_log((jnp.float64(1.0) + root_squared) * jnp.float64(0.5))


def monin_obukhov_initialize(
    reference_wind_speed: jnp.ndarray,
    virtual_potential_temperature: jnp.ndarray,
    virtual_potential_temperature_difference: jnp.ndarray,
    reference_height_minus_displacement: jnp.ndarray,
    momentum_roughness_length: jnp.ndarray,
    maximum_stable_zeta: float | jnp.ndarray = 0.5,
) -> MoninObukhovInitialization:

    wind = jnp.asarray(reference_wind_speed, dtype=jnp.float64)
    virtual_temperature = jnp.asarray(virtual_potential_temperature, dtype=jnp.float64)
    difference = jnp.asarray(virtual_potential_temperature_difference, dtype=jnp.float64)
    height = jnp.asarray(reference_height_minus_displacement, dtype=jnp.float64)
    roughness = jnp.asarray(momentum_roughness_length, dtype=jnp.float64)
    maximum_stable = jnp.asarray(maximum_stable_zeta, dtype=jnp.float64)
    wind, virtual_temperature, difference, height, roughness = jnp.broadcast_arrays(
        wind,
        virtual_temperature,
        difference,
        height,
        roughness,
    )

    stable_wind = jnp.maximum(wind, jnp.float64(0.1))
    unstable_wind = jnp.sqrt(
        fortran_add(
            fortran_product(wind, wind),
            jnp.full_like(wind, jnp.float64(0.5) * jnp.float64(0.5)),
        )
    )
    stability_wind = jnp.where(difference >= jnp.float64(0.0), stable_wind, unstable_wind)
    richardson = fortran_divide(
        fortran_product(
            fortran_product(jnp.float64(GRAVITY), height),
            difference,
        ),
        fortran_product(
            fortran_product(virtual_temperature, stability_wind),
            stability_wind,
        ),
    )
    log_height = _elementwise_log(fortran_divide(height, roughness))
    stable_zeta = fortran_divide(
        fortran_product(richardson, log_height),
        fortran_add(
            jnp.ones_like(richardson),
            -fortran_product(
                jnp.float64(5.0),
                jnp.minimum(richardson, jnp.float64(0.19)),
            ),
        ),
    )
    stable_zeta = jnp.minimum(
        maximum_stable,
        jnp.maximum(stable_zeta, jnp.float64(0.01)),
    )
    unstable_zeta = fortran_product(richardson, log_height)
    unstable_zeta = jnp.maximum(
        jnp.float64(-100.0),
        jnp.minimum(unstable_zeta, jnp.float64(-0.01)),
    )
    zeta = jnp.where(richardson >= jnp.float64(0.0), stable_zeta, unstable_zeta)
    return MoninObukhovInitialization(
        stability_wind,
        fortran_divide(height, zeta),
    )


def _momentum_friction_velocity(
    height: jnp.ndarray,
    roughness: jnp.ndarray,
    monin_obukhov_length: jnp.ndarray,
    stability_wind: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    transition = jnp.float64(MOMENTUM_TRANSITION)
    zeta = fortran_divide(height, monin_obukhov_length)
    unstable_length = jnp.where(
        monin_obukhov_length < jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(-1.0),
    )
    stable_length = jnp.where(
        monin_obukhov_length > jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(1.0),
    )
    unstable_zeta = jnp.where(zeta < 0.0, zeta, jnp.float64(-0.01))
    stable_zeta = jnp.where(zeta >= 0.0, zeta, jnp.float64(0.01))

    def add_terms(*terms: jnp.ndarray) -> jnp.ndarray:
        result = terms[0]
        for term in terms[1:]:
            result = fortran_add(result, term)
        return result

    strong_unstable_correction = fortran_product(
        jnp.float64(1.14),
        fortran_add(
            _stability_power(-unstable_zeta, 0.333),
            -(transition ** jnp.float64(0.333)),
        ),
    )
    strong_unstable = add_terms(
        _elementwise_log(
            fortran_divide(
                fortran_product(-transition, unstable_length),
                roughness,
            )
        ),
        -stability_function_momentum(-transition),
        stability_function_momentum(fortran_divide(roughness, unstable_length)),
        strong_unstable_correction,
    )
    unstable = add_terms(
        _elementwise_log(fortran_divide(height, roughness)),
        -stability_function_momentum(unstable_zeta),
        stability_function_momentum(fortran_divide(roughness, unstable_length)),
    )
    weak_stable = add_terms(
        _elementwise_log(fortran_divide(height, roughness)),
        fortran_product(jnp.float64(5.0), stable_zeta),
        -fortran_divide(
            fortran_product(jnp.float64(5.0), roughness),
            stable_length,
        ),
    )
    strong_stable = _strong_stable_denominator(
        stable_length,
        roughness,
        stable_zeta,
    )
    denominator = jnp.where(
        zeta < -transition,
        strong_unstable,
        jnp.where(
            zeta < jnp.float64(0.0),
            unstable,
            jnp.where(zeta <= jnp.float64(1.0), weak_stable, strong_stable),
        ),
    )
    numerator = fortran_product(jnp.float64(VON_KARMAN), stability_wind)
    return fortran_divide(numerator, denominator), zeta, numerator, denominator


def _scalar_profile(
    height: jnp.ndarray,
    roughness: jnp.ndarray,
    monin_obukhov_length: jnp.ndarray,
) -> jnp.ndarray:
    transition = jnp.float64(SCALAR_TRANSITION)
    zeta = fortran_divide(height, monin_obukhov_length)
    unstable_length = jnp.where(
        monin_obukhov_length < jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(-1.0),
    )
    stable_length = jnp.where(
        monin_obukhov_length > jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(1.0),
    )
    unstable_zeta = jnp.where(zeta < 0.0, zeta, jnp.float64(-0.01))
    stable_zeta = jnp.where(zeta >= 0.0, zeta, jnp.float64(0.01))

    def add_terms(*terms: jnp.ndarray) -> jnp.ndarray:
        result = terms[0]
        for term in terms[1:]:
            result = fortran_add(result, term)
        return result

    strong_unstable_correction = fortran_product(
        jnp.float64(0.8),
        fortran_add(
            transition ** jnp.float64(-0.333),
            -_stability_power(-unstable_zeta, -0.333),
        ),
    )
    strong_unstable = add_terms(
        _elementwise_log(
            fortran_divide(
                fortran_product(-transition, unstable_length),
                roughness,
            )
        ),
        -stability_function_scalar(-transition),
        stability_function_scalar(fortran_divide(roughness, unstable_length)),
        strong_unstable_correction,
    )
    unstable = add_terms(
        _elementwise_log(fortran_divide(height, roughness)),
        -stability_function_scalar(unstable_zeta),
        stability_function_scalar(fortran_divide(roughness, unstable_length)),
    )
    weak_stable = add_terms(
        _elementwise_log(fortran_divide(height, roughness)),
        fortran_product(jnp.float64(5.0), stable_zeta),
        -fortran_divide(
            fortran_product(jnp.float64(5.0), roughness),
            stable_length,
        ),
    )
    strong_stable = _strong_stable_denominator(
        stable_length,
        roughness,
        stable_zeta,
    )
    denominator = jnp.where(
        zeta < -transition,
        strong_unstable,
        jnp.where(
            zeta < jnp.float64(0.0),
            unstable,
            jnp.where(zeta <= jnp.float64(1.0), weak_stable, strong_stable),
        ),
    )
    return _refined_divide(
        jnp.full_like(denominator, jnp.float64(VON_KARMAN)),
        denominator,
    )


def _wind_at_ten_meters(
    height: jnp.ndarray,
    roughness: jnp.ndarray,
    monin_obukhov_length: jnp.ndarray,
    stability_wind: jnp.ndarray,
    friction_velocity_value: jnp.ndarray,
    zeta: jnp.ndarray,
) -> jnp.ndarray:
    transition = jnp.float64(MOMENTUM_TRANSITION)
    unstable_length = jnp.where(
        monin_obukhov_length < jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(-1.0),
    )
    stable_length = jnp.where(
        monin_obukhov_length > jnp.float64(0.0),
        monin_obukhov_length,
        jnp.float64(1.0),
    )
    unstable_zeta = jnp.where(zeta < 0.0, zeta, jnp.float64(-0.01))
    stable_zeta = jnp.where(zeta >= 0.0, zeta, jnp.float64(0.01))
    target_height = fortran_add(jnp.float64(10.0), roughness)

    def add_terms(*terms: jnp.ndarray) -> jnp.ndarray:
        result = terms[0]
        for term in terms[1:]:
            result = fortran_add(result, term)
        return result

    strong_unstable_correction = fortran_product(
        jnp.float64(1.14),
        fortran_add(
            _stability_power(-unstable_zeta, 0.333),
            -(transition ** jnp.float64(0.333)),
        ),
    )
    strong_unstable = add_terms(
        _elementwise_log(
            fortran_divide(
                fortran_product(-transition, unstable_length),
                target_height,
            )
        ),
        -stability_function_momentum(-transition),
        stability_function_momentum(fortran_divide(target_height, unstable_length)),
        strong_unstable_correction,
    )
    unstable = add_terms(
        _elementwise_log(fortran_divide(height, target_height)),
        -stability_function_momentum(unstable_zeta),
        stability_function_momentum(fortran_divide(target_height, unstable_length)),
    )
    weak_stable = add_terms(
        _elementwise_log(fortran_divide(height, target_height)),
        fortran_product(jnp.float64(5.0), stable_zeta),
        -fortran_divide(
            fortran_product(jnp.float64(5.0), target_height),
            stable_length,
        ),
    )
    strong_stable_final_group = add_terms(
        fortran_product(jnp.float64(5.0), _elementwise_log(stable_zeta)),
        stable_zeta,
        jnp.float64(-1.0),
    )
    strong_stable = add_terms(
        _elementwise_log(fortran_divide(stable_length, target_height)),
        jnp.float64(5.0),
        -fortran_divide(
            fortran_product(jnp.float64(5.0), target_height),
            stable_length,
        ),
        strong_stable_final_group,
    )
    profile_difference = jnp.where(
        zeta < -transition,
        strong_unstable,
        jnp.where(
            zeta < jnp.float64(0.0),
            unstable,
            jnp.where(zeta <= jnp.float64(1.0), weak_stable, strong_stable),
        ),
    )
    correction = fortran_product(
        _refined_divide(friction_velocity_value, jnp.float64(VON_KARMAN)),
        profile_difference,
    )
    diagnosed = fortran_add(stability_wind, -correction)
    return jnp.where(
        fortran_add(height, -roughness) <= jnp.float64(10.0),
        stability_wind,
        diagnosed,
    )


def friction_velocity(
    active_patch: jnp.ndarray,
    displacement_height_patch: jnp.ndarray,
    momentum_roughness_patch: jnp.ndarray,
    sensible_roughness_patch: jnp.ndarray,
    latent_roughness_patch: jnp.ndarray,
    monin_obukhov_length_patch: jnp.ndarray,
    iteration: int | jnp.ndarray,
    reference_wind_speed_patch: jnp.ndarray,
    stability_wind_speed_patch: jnp.ndarray,
    forcing_height_wind_patch: jnp.ndarray,
    forcing_height_temperature_patch: jnp.ndarray,
    forcing_height_humidity_patch: jnp.ndarray,
    state: FrictionVelocityState,
) -> FrictionVelocityResult:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    size = active.shape[0]
    arrays = {
        "displacement_height_patch": displacement_height_patch,
        "momentum_roughness_patch": momentum_roughness_patch,
        "sensible_roughness_patch": sensible_roughness_patch,
        "latent_roughness_patch": latent_roughness_patch,
        "monin_obukhov_length_patch": monin_obukhov_length_patch,
        "reference_wind_speed_patch": reference_wind_speed_patch,
        "stability_wind_speed_patch": stability_wind_speed_patch,
        "forcing_height_wind_patch": forcing_height_wind_patch,
        "forcing_height_temperature_patch": forcing_height_temperature_patch,
        "forcing_height_humidity_patch": forcing_height_humidity_patch,
    }
    arrays = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in arrays.items()}
    _validate_vector_shapes(size, tuple(arrays.items()))
    _validate_vector_shapes(
        size,
        tuple(zip(FrictionVelocityState._fields, state, strict=True)),
    )

    displacement = jnp.where(active, arrays["displacement_height_patch"], 0.0)
    z0m = jnp.where(active, arrays["momentum_roughness_patch"], 0.01)
    z0h = jnp.where(active, arrays["sensible_roughness_patch"], 0.01)
    z0q = jnp.where(active, arrays["latent_roughness_patch"], 0.01)
    obu = jnp.where(active, arrays["monin_obukhov_length_patch"], 100.0)
    wind = jnp.where(active, arrays["reference_wind_speed_patch"], 1.0)
    stability_wind = jnp.where(active, arrays["stability_wind_speed_patch"], 1.0)
    wind_height = jnp.where(active, arrays["forcing_height_wind_patch"], 30.0)
    temperature_height = jnp.where(active, arrays["forcing_height_temperature_patch"], 30.0)
    humidity_height = jnp.where(active, arrays["forcing_height_humidity_patch"], 30.0)

    wind_height_minus_displacement = wind_height - displacement
    friction, zeta, momentum_numerator, momentum_denominator = _momentum_friction_velocity(
        wind_height_minus_displacement,
        z0m,
        obu,
        stability_wind,
    )
    deposition = jnp.where(
        zeta < 0.0,
        jnp.float64(2.0e-3)
        * friction
        * (jnp.float64(1.0) + (jnp.float64(300.0) / (-obu)) ** jnp.float64(0.666)),
        jnp.float64(2.0e-3) * friction,
    )
    wind_10m = _wind_at_ten_meters(
        wind_height_minus_displacement,
        z0m,
        obu,
        stability_wind,
        friction,
        zeta,
    )
    temperature_profile = _scalar_profile(temperature_height - displacement, z0h, obu)
    calculated_humidity_profile = _scalar_profile(humidity_height - displacement, z0q, obu)
    humidity_profile = jnp.where(
        (humidity_height == temperature_height) & (z0q == z0h),
        temperature_profile,
        calculated_humidity_profile,
    )
    temperature_profile_2m = _scalar_profile(jnp.float64(2.0) + z0h, z0h, obu)
    calculated_humidity_profile_2m = _scalar_profile(jnp.float64(2.0) + z0q, z0q, obu)
    humidity_profile_2m = jnp.where(
        z0q == z0h,
        temperature_profile_2m,
        calculated_humidity_profile_2m,
    )

    dust_zeta = jnp.minimum(zeta, jnp.float64(1.0))
    unstable_dust_zeta = jnp.where(dust_zeta < 0.0, dust_zeta, jnp.float64(-0.01))
    dust_root = _stability_power(
        fortran_add(jnp.float64(1.0), -fortran_product(jnp.float64(16.0), unstable_dust_zeta)),
        0.25,
    )
    dust_tmp2 = _elementwise_log(
        fortran_add(
            jnp.float64(1.0),
            fortran_product(dust_root, dust_root),
        )
        / jnp.float64(2.0)
    )
    dust_tmp3 = _elementwise_log(fortran_add(jnp.float64(1.0), dust_root) / jnp.float64(2.0))
    dust_fm_unstable = fortran_ordered_sum(
        jnp.stack(
            (
                fortran_product(jnp.float64(2.0), dust_tmp3),
                dust_tmp2,
                -fortran_product(jnp.float64(2.0), _elementwise_atan(dust_root)),
                jnp.full_like(dust_root, 1.5707963),
            ),
            axis=0,
        )
    )
    dust_fm_new = jnp.where(
        dust_zeta < 0.0,
        dust_fm_unstable,
        -jnp.float64(5.0) * dust_zeta,
    )
    previous_fm = jnp.asarray(state.momentum_stability_patch, dtype=jnp.float64)
    averaged_dust_fm = fortran_rounding_barrier(
        fortran_product(
            jnp.float64(0.5),
            fortran_rounding_barrier(previous_fm + dust_fm_new),
        )
    )
    dust_fm = jnp.where(
        jnp.asarray(iteration, dtype=jnp.int32) == jnp.int32(1),
        dust_fm_new,
        averaged_dust_fm,
    )
    zeta_10m = jnp.minimum(jnp.float64(10.0) / obu, jnp.float64(1.0))
    zeta_10m = jnp.where(zeta == jnp.float64(0.0), jnp.float64(0.0), zeta_10m)
    unstable_zeta_10m = jnp.where(zeta_10m < 0.0, zeta_10m, jnp.float64(-0.01))
    dust_root_10m = _stability_power(
        fortran_add(jnp.float64(1.0), -fortran_product(jnp.float64(16.0), unstable_zeta_10m)),
        0.25,
    )
    dust_tmp2_10m = _elementwise_log(
        fortran_add(
            jnp.float64(1.0),
            fortran_product(dust_root_10m, dust_root_10m),
        )
        / jnp.float64(2.0)
    )
    dust_tmp3_10m = _elementwise_log(
        fortran_add(jnp.float64(1.0), dust_root_10m) / jnp.float64(2.0)
    )
    dust_fm_10m_unstable = fortran_ordered_sum(
        jnp.stack(
            (
                fortran_product(jnp.float64(2.0), dust_tmp3_10m),
                dust_tmp2_10m,
                -fortran_product(jnp.float64(2.0), _elementwise_atan(dust_root_10m)),
                jnp.full_like(dust_root_10m, 1.5707963),
            ),
            axis=0,
        )
    )
    dust_fm_10m = jnp.where(
        zeta_10m < 0.0,
        dust_fm_10m_unstable,
        -jnp.float64(5.0) * zeta_10m,
    )
    height_ratio = lax.optimization_barrier(
        _refined_divide(wind_height, jnp.full_like(wind_height, 10.0))
    )
    height_log = _elementwise_log(jnp.maximum(jnp.float64(1.0), height_ratio))
    dust_profile_difference = fortran_ordered_sum(
        jnp.stack((height_log, -dust_fm, dust_fm_10m), axis=0)
    )
    dust_correction = _multiply_boundary(
        friction / jnp.float64(VON_KARMAN),
        dust_profile_difference,
    )
    dust_correction = lax.optimization_barrier(dust_correction)
    wind_10m_dust = lax.optimization_barrier(
        fortran_ordered_sum(jnp.stack((wind, -dust_correction), axis=0))
    )

    def update(value: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, value, jnp.asarray(previous, dtype=jnp.float64))

    public_state = FrictionVelocityState(
        momentum_stability_patch=update(dust_fm, state.momentum_stability_patch),
        deposition_velocity_patch=update(deposition, state.deposition_velocity_patch),
        wind_10m_dust_patch=update(wind_10m_dust, state.wind_10m_dust_patch),
        wind_10m_patch=update(wind_10m, state.wind_10m_patch),
        atmospheric_wind_patch=update(stability_wind, state.atmospheric_wind_patch),
        dust_friction_velocity_patch=update(friction, state.dust_friction_velocity_patch),
    )
    return FrictionVelocityResult(
        state=public_state,
        friction_velocity_patch=jnp.where(active, friction, 0.0),
        temperature_profile_patch=jnp.where(active, temperature_profile, 0.0),
        humidity_profile_patch=jnp.where(active, humidity_profile, 0.0),
        temperature_profile_2m_patch=jnp.where(active, temperature_profile_2m, 0.0),
        humidity_profile_2m_patch=jnp.where(active, humidity_profile_2m, 0.0),
        momentum_numerator_patch=jnp.where(active, momentum_numerator, 0.0),
        momentum_denominator_patch=jnp.where(active, momentum_denominator, 1.0),
        momentum_zeta_patch=jnp.where(active, zeta, 0.0),
    )


import jax

_native_friction_velocity = friction_velocity


@jax.custom_jvp
def friction_velocity(
    active_patch,
    displacement_height_patch,
    momentum_roughness_patch,
    sensible_roughness_patch,
    latent_roughness_patch,
    monin_obukhov_length_patch,
    iteration,
    reference_wind_speed_patch,
    stability_wind_speed_patch,
    forcing_height_wind_patch,
    forcing_height_temperature_patch,
    forcing_height_humidity_patch,
    state,
):
    return _native_friction_velocity(
        active_patch,
        displacement_height_patch,
        momentum_roughness_patch,
        sensible_roughness_patch,
        latent_roughness_patch,
        monin_obukhov_length_patch,
        iteration,
        reference_wind_speed_patch,
        stability_wind_speed_patch,
        forcing_height_wind_patch,
        forcing_height_temperature_patch,
        forcing_height_humidity_patch,
        state,
    )


@friction_velocity.defjvp
def _friction_velocity_deposition_jvp(primals, tangents):
    native, dot = jax.jvp(_native_friction_velocity, primals, tangents)
    active = jnp.asarray(primals[0], dtype=jnp.bool_)
    obu = jnp.where(active, primals[5], jnp.float64(100.0))
    obu_dot = jnp.where(active, tangents[5], jnp.float64(0.0))
    unstable = native.momentum_zeta_patch < jnp.float64(0.0)
    derivative_obu = jnp.where(unstable, obu, jnp.float64(-1.0))
    power = (jnp.float64(300.0) / (-derivative_obu)) ** jnp.float64(0.666)
    coefficient = jnp.where(
        unstable,
        jnp.float64(2.0e-3) * (jnp.float64(1.0) + power),
        jnp.float64(2.0e-3),
    )
    length_coefficient = jnp.where(
        unstable,
        -jnp.float64(2.0e-3) * jnp.float64(0.666)
        * native.friction_velocity_patch * power / derivative_obu,
        jnp.float64(0.0),
    )
    deposition_dot = jnp.where(
        active,
        coefficient * dot.friction_velocity_patch + length_coefficient * obu_dot,
        tangents[12].deposition_velocity_patch,
    )
    return native, dot._replace(
        state=dot.state._replace(deposition_velocity_patch=deposition_dot)
    )
