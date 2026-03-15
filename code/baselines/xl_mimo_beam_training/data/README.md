# Data Preparation

## Required Files

Place the following `.mat` files in this directory (or specify path via `data_path` in config):

- `pcsi.mat` — Perfect Channel State Information (CSI)
  - Variable name: `pcsi`
  - Shape: `(num_samples, N_t)` complex-valued
- `ecsi.mat` — Estimated CSI (with channel estimation errors)
  - Variable name: `ecsi`
  - Shape: `(num_samples, N_t)` complex-valued

## Data Format

Each row represents one channel realization between the XL-MIMO base station
and a user. The channel is modeled using the spherical wave near-field model
described in the paper.

## Synthetic Data

If real data is not available, the code automatically generates synthetic
near-field channel data using the spherical wave model. See `generate_synthetic_data()` in `src/utils.py`.

## Download

Data is generated using the MATLAB scripts from the original repository:
https://github.com/yuanhao-cui/near-field-beamforming-using-deeplearning

Run the MATLAB channel generation scripts first, then transfer the `.mat` files here.
