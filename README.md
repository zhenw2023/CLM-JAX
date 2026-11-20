# CLM-JAX

[English](README.md) | [简体中文](README.ZH.md)

CLM-JAX implements the CLM5.0 BGC-Crop land model in JAX, with double-precision
CPU and NVIDIA GPU execution and automatic differentiation. This release uses
the CESM2.1.3 `I2000Clm50BgcCrop` configuration and includes WFDE5-to-f09 mapping
weights and the model parameter, SNICAR and MEGAN lookup tables.

## 1. Install

From this directory, create a Python 3.11 environment with Mamba.

CPU:

```bash
mamba env create -f environment.yml
mamba activate clm-jax
python -m pip install --no-deps --no-build-isolation -e .
```

NVIDIA GPU (Linux x86_64, CUDA 12.9):

```bash
mamba env create -f environment-gpu.yml
mamba activate clm-jax-gpu
python -m pip install --no-deps --no-build-isolation -e .
```

## 2. Prepare inputs

Use a prepared CESM2.1.3 `I2000Clm50BgcCrop` case on the f09 land grid, with
WFDE5 forcing. Keep these inputs in your data directory:

- CIME case files: `env_*.xml`, generated namelists and DATM streams in
  `CaseDocs/`, and input lists in `Buildconf/`.
- The case's domain, surface and forcing datasets, including the meteorology,
  solar and precipitation streams for the simulation period.
- Land (`clm2.r`) and coupler (`cpl.r`) restart files for the start time.

Set the paths in the case XML, namelists, input lists and stream files to your
local datasets. Point the following namelist entries to the bundled files
under this release's `inputdata/`:

| Namelist entry | Bundled file |
| --- | --- |
| `lnd_in`: `paramfile` | `lnd/clm2/paramdata/clm5_params.c171117.nc` |
| `lnd_in`: `fsnowaging` | `lnd/clm2/snicardata/snicar_drdt_bst_fit_60_c070416.nc` |
| `lnd_in`: `fsnowoptics` | `lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc` |
| `drv_flds_in`: `megan_factors_file` | `atm/cam/chem/trop_mozart/emis/megan21_emis_factors_78pft_c20161108.nc` |

The case's `DIN_LOC_ROOT` and `config.toml`'s `case.inputdata_root` use the same
external inputdata directory.

## 3. Run

Edit [`config.toml`](config.toml): set the four `[case]` paths, simulation start
and stop times, and `execution.backend` to `"cpu"` or `"gpu"`. Match the start
time and time step to the restart files, and set `maximum_rss_gib` to the host
memory budget. The supplied example advances one day with 1800-second steps.

```bash
python main.py
```

Run settings are read from `config.toml` next to `main.py`. The model writes
gridcell and PFT NetCDF history files to `../clm-jax-output/` and stores its JAX
compilation cache in `../clm-jax-cache/`. Set the output interval and directory
in `[output]`.

## License

[LGPL-2.1-or-later](LICENSE).
