# SharpaWave dex-retargeting data

- `*_sharpa_wave_with_flange.urdf` — vendored unmodified from
  [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
  (`wave_01/{left,right}_sharpa_wave/`), fetched 2026-07-30. Used for kinematics
  only (dex_retargeting/pinocchio); the `package://` mesh references are expected
  to be unresolvable and are harmless. Same source and vintage as the USD assets
  vendored in the SharpaWave RoboLab fork (`assets/robots/sharpa_wave/config.yaml`).
- `sharpa_wave_{left,right}_dexpilot.yml` — DexPilot retargeting configs
  (`dex_retargeting` schema, mirroring Isaac Lab's Fourier-hand configs):
  5 fingertip task links, all 22 actuated joints as targets. `urdf_path` is
  resolved relative to the parent directory at load time and never rewritten
  on disk.
