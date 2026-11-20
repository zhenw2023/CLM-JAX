
from functools import wraps

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core

from clm_jax.floating_point import correctly_rounded_divide
from clm_jax.physics.numerics import fortran_hardware_add, fortran_hardware_multiply
from clm_jax.physics.solar import raw_cosine_solar_zenith, solar_declination


def _rounded_program(program, arguments, *, native_trigonometry=False):
    environment = dict(zip(program.jaxpr.invars, arguments, strict=True))
    environment.update(zip(program.jaxpr.constvars, program.consts, strict=True))

    def read(variable):
        return variable.val if isinstance(variable, core.Literal) else environment[variable]

    arithmetic = {
        lax.add_p: fortran_hardware_add,
        lax.sub_p: lambda left, right: fortran_hardware_add(left, -right),
        lax.mul_p: fortran_hardware_multiply,
        lax.div_p: correctly_rounded_divide,
    }
    if native_trigonometry:
        from clm_jax.physics.native_solar_math import (
            native_asin, native_cos, native_sin, native_sincos_cos, native_sincos_sin,
        )
        arithmetic.update({lax.sin_p: native_sin, lax.cos_p: native_cos,
                           lax.asin_p: native_asin})
        sine_operands = {e.invars[0] for e in program.jaxpr.eqns
                         if e.primitive == lax.sin_p and not isinstance(e.invars[0], core.Literal)}
        cosine_operands = {e.invars[0] for e in program.jaxpr.eqns
                           if e.primitive == lax.cos_p and not isinstance(e.invars[0], core.Literal)}
        paired_trig = sine_operands.intersection(cosine_operands)
    for equation in program.jaxpr.eqns:
        primitive = equation.primitive
        values = [read(variable) for variable in equation.invars]
        if (native_trigonometry and primitive in (lax.sin_p, lax.cos_p)
                and not isinstance(equation.invars[0], core.Literal)
                and equation.invars[0] in paired_trig):
            function = native_sincos_sin if primitive == lax.sin_p else native_sincos_cos
            result = [function(*values)]
        elif primitive in arithmetic:
            result = [arithmetic[primitive](*values)]
        else:
            if primitive in (lax.sin_p, lax.cos_p, lax.asin_p):
                values = lax.optimization_barrier(tuple(
                    jnp.asarray(value, dtype=jnp.float64) for value in values
                ))
            result = primitive.bind(*values, **equation.params)
            if not primitive.multiple_results:
                result = [result]
        environment.update(zip(equation.outvars, result, strict=True))
    return [read(variable) for variable in program.jaxpr.outvars]


def _compile_solar(function, *, native_trigonometry=False):
    @jax.custom_jvp
    def rounded(*arguments):
        program, shape = jax.make_jaxpr(function, return_shape=True)(*arguments)
        return jax.tree.structure(shape).unflatten(
            _rounded_program(program, jax.tree.leaves(arguments),
                             native_trigonometry=native_trigonometry)
        )

    @rounded.defjvp
    def derivative(primals, tangents):
        return rounded(*primals), jax.jvp(function, primals, tangents)[1]

    compiled = jax.jit(rounded)

    @wraps(function)
    def execute(*arguments):
        return (compiled(*arguments) if native_trigonometry or jax.default_backend() == "gpu"
                else function(*arguments))

    return execute


def native_grid_radians(degrees):
    from clm_jax.constants import PI
    return correctly_rounded_divide(
        fortran_hardware_multiply(jnp.asarray(degrees, jnp.float64), jnp.float64(PI)),
        jnp.float64(180.),
    )


solar_declination_runtime = _compile_solar(solar_declination)
raw_cosine_solar_zenith_runtime = _compile_solar(raw_cosine_solar_zenith)

solar_declination_native_runtime = _compile_solar(solar_declination, native_trigonometry=True)
raw_cosine_solar_zenith_native_runtime = _compile_solar(raw_cosine_solar_zenith, native_trigonometry=True)


@jax.custom_jvp
def _datm_sample_day(lower_day, elapsed_seconds):
    lower_day = jnp.asarray(lower_day, jnp.float64)
    day = jnp.floor(lower_day)
    seconds = jnp.rint((lower_day - day) * jnp.float64(86400.)) + elapsed_seconds
    advanced_day = day + jnp.floor(seconds / jnp.float64(86400.))
    time_of_day = jnp.mod(seconds, jnp.float64(86400.))
    advanced_day = jnp.mod(advanced_day - jnp.float64(1.), jnp.float64(365.)) + 1.
    return fortran_hardware_add(
        advanced_day, correctly_rounded_divide(time_of_day, jnp.float64(86400.)),
    )


@_datm_sample_day.defjvp
def _datm_sample_day_jvp(primals, tangents):
    lower_day, elapsed = primals
    lower_tangent, elapsed_tangent = tangents
    return (_datm_sample_day(lower_day, elapsed),
            lower_tangent + elapsed_tangent / jnp.float64(86400.))


def native_datm_coszen_temporal_interpolate(
    interval_mean_shortwave, model_calendar_day, lower_calendar_day,
    latitude_radians, longitude_radians, *, interval_seconds, model_step_seconds,
    orbit=None,
):
    from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT
    if interval_seconds <= 0 or model_step_seconds <= 0:
        raise ValueError("forcing interval and model step must be positive")
    orbit = FIXED_YEAR_2000_ORBIT if orbit is None else orbit
    local_step = model_step_seconds
    if interval_seconds % local_step:
        local_step = interval_seconds // (interval_seconds // local_step + 1)

    def cosine(day):
        return jnp.maximum(jnp.float64(.001), raw_cosine_solar_zenith_native_runtime(
            day, latitude_radians, longitude_radians, orbit,
        ))

    total = jnp.zeros_like(jnp.broadcast_arrays(latitude_radians, longitude_radians)[0],
                           dtype=jnp.float64)
    elapsed = 0
    while elapsed < interval_seconds:
        step = min(local_step, interval_seconds - elapsed)
        elapsed += step
        sample = _datm_sample_day(lower_calendar_day, jnp.float64(elapsed))
        total = fortran_hardware_add(total, fortran_hardware_multiply(
            cosine(sample), jnp.float64(step),
        ))
    average = correctly_rounded_divide(total, jnp.float64(interval_seconds))
    instantaneous = cosine(model_calendar_day)
    reconstructed = correctly_rounded_divide(fortran_hardware_multiply(
        jnp.asarray(interval_mean_shortwave, jnp.float64), instantaneous,
    ), average)
    return jnp.where(instantaneous > jnp.float64(.001), reconstructed, jnp.float64(0.))
