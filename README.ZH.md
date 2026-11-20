# CLM-JAX

[English](README.md) | [简体中文](README.ZH.md)

CLM-JAX 使用 JAX 实现 CLM5.0 BGC-Crop 陆面模式，支持 CPU 和 NVIDIA GPU 双精度计算及自动微分。本版本采用 CESM2.1.3 的 `I2000Clm50BgcCrop` 配置，附带 WFDE5 到 f09 格点的映射权重、模式参数表及 SNICAR、MEGAN 查找表。

## 1. 安装环境

在本目录下使用 Mamba 创建 Python 3.11 环境。

CPU：

```bash
mamba env create -f environment.yml
mamba activate clm-jax
python -m pip install --no-deps --no-build-isolation -e .
```

NVIDIA GPU（Linux x86_64，CUDA 12.9）：

```bash
mamba env create -f environment-gpu.yml
mamba activate clm-jax-gpu
python -m pip install --no-deps --no-build-isolation -e .
```

## 2. 准备输入

使用已准备好的 CESM2.1.3 `I2000Clm50BgcCrop` 算例，采用 f09 陆面网格和 WFDE5 强迫。将以下输入放在自己的数据目录中：

- CIME 算例文件：`env_*.xml`、`CaseDocs/` 中生成的 namelist 和 DATM stream 文件，以及 `Buildconf/` 中的输入文件清单。
- 算例所需的网格、地表及强迫数据，包括覆盖模拟时段的气象、短波辐射和降水流。
- 与起始时刻对应的陆面重启文件（`clm2.r`）和耦合器重启文件（`cpl.r`）。

将算例 XML、namelist、输入清单和 stream 中的路径设置为本地数据路径。将以下 namelist 项指向本版本 `inputdata/` 下的对应文件：

| Namelist 项 | 附带文件 |
| --- | --- |
| `lnd_in`：`paramfile` | `lnd/clm2/paramdata/clm5_params.c171117.nc` |
| `lnd_in`：`fsnowaging` | `lnd/clm2/snicardata/snicar_drdt_bst_fit_60_c070416.nc` |
| `lnd_in`：`fsnowoptics` | `lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc` |
| `drv_flds_in`：`megan_factors_file` | `atm/cam/chem/trop_mozart/emis/megan21_emis_factors_78pft_c20161108.nc` |

算例的 `DIN_LOC_ROOT` 与 `config.toml` 中的 `case.inputdata_root` 填写同一个外部输入数据目录。

## 3. 运行

编辑 [`config.toml`](config.toml)，填写 `[case]` 的四个路径、模拟起止时间，并将 `execution.backend` 设为 `"cpu"` 或 `"gpu"`。起始时刻和时间步长与重启文件保持一致，`maximum_rss_gib` 按主机内存预算设置。示例配置以 1800 秒为步长积分一天。

```bash
python main.py
```

运行参数由 `main.py` 同目录下的 `config.toml` 读取。模式将格点和 PFT 的 NetCDF 历史输出写入 `../clm-jax-output/`，JAX 编译缓存写入 `../clm-jax-cache/`。输出间隔和目录在 `[output]` 中设置。

## 许可证

[LGPL-2.1-or-later](LICENSE)。
