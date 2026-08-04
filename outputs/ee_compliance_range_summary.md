# EE Compliance Range Eval Summary

All four policies below were evaluated with the same target stiffness:
`[kx, ky, kz] = [300, 450, 500] N/m`.

| Policy | Nominal err (m) | **Compliance err (m)** | Measured stiffness x (N/m) | Measured stiffness y (N/m) | Measured stiffness z (N/m) | Stiffness MAE x (N/m) | Stiffness MAE y (N/m) | Stiffness MAE z (N/m) | **Overall stiffness MAE (N/m)** | Force pred/actual norm (N) | Force error norm mean±std/rmse/max (N) | Force error xyz mean±std (N) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| range_v1 | 0.0421 ± 0.0307 | **0.0277 ± 0.0153** | 278.0 ± 52.8 | 320.8 ± 57.1 | 299.6 ± 105.1 | 48.9 ± 29.7 | 131.6 ± 51.3 | 200.4 ± 105.1 | **127.0 ± 93.2** | 14.9 ± 8.7 / 16.0 ± 8.6 | 2.9 ± 1.8 / 3.4 / 11.6 | x=0.5 ± 2.3, y=-0.3 ± 1.2, z=-0.9 ± 1.9 |
| analytical_moe | 0.0368 ± 0.0245 | **0.0299 ± 0.0131** | 385.0 ± 200.1 | 526.1 ± 129.8 | 547.4 ± 121.0 | 120.8 ± 180.7 | 128.6 ± 78.1 | 99.9 ± 83.1 | **116.4 ± 124.0** | 15.1 ± 9.0 / 16.0 ± 8.6 | 6.2 ± 3.5 / 7.1 / 17.2 | x=-2.2 ± 4.5, y=2.3 ± 1.8, z=1.2 ± 3.9 |
| learned_moe_gate | 0.0417 ± 0.0146 | **0.0395 ± 0.0104** | 614.0 ± 257.2 | 848.2 ± 511.5 | 521.2 ± 114.5 | 314.0 ± 257.2 | 416.3 ± 496.9 | 97.9 ± 63.0 | **276.1 ± 351.1** | 11.6 ± 4.9 / 16.0 ± 8.6 | 20.1 ± 9.2 / 22.1 / 45.2 | x=-9.4 ± 11.7, y=-3.8 ± 10.5, z=2.6 ± 11.5 |
| range_v2 | 0.0403 ± 0.0215 | **0.0301 ± 0.0112** | 384.4 ± 149.7 | 444.0 ± 93.0 | 384.3 ± 69.7 | 93.8 ± 144.0 | 69.9 ± 61.6 | 124.2 ± 53.2 | **95.9 ± 98.1** | 15.6 ± 8.9 / 16.0 ± 8.6 | 1.9 ± 0.9 / 2.1 / 4.1 | x=0.3 ± 1.2, y=-0.3 ± 1.0, z=-0.4 ± 1.2 |

## Reports

- `range_v1`: `outputs/ee_compliance_eval_ee_xyz_range_200_600_3kp_force_b_stu_adapt_20260802_121433_k300_450_500.json`
- `analytical_moe`: `outputs/ee_compliance_eval_G1_ee_analytical_200_600_k300_450_500.json`
- `learned_moe_gate`: `outputs/ee_compliance_eval_ee_xyz_learned_moe_200_600_gate_20260802_192147_k300_450_500.json`
- `range_v2`: `outputs/ee_compliance_eval_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k300_450_500.json`

## V1 Versus V2

- Nominal tracking improved: mean error decreased from `0.0421 m` to `0.0403 m`, and std decreased from `0.0307 m` to `0.0215 m`.
- Compliance error did not improve in mean: `0.0277 m` to `0.0301 m`; RMSE was nearly unchanged (`0.0317 m` to `0.0321 m`).
- Overall stiffness MAE improved from `127.0` to `95.9 N/m`.
- Axis-wise stiffness MAE improved on y and z, but worsened on x: x `48.9 -> 93.8`, y `131.6 -> 69.9`, z `200.4 -> 124.2 N/m`.
- Force estimator improved substantially: error-norm mean `2.9 -> 1.9 N`, RMSE `3.4 -> 2.1 N`, and maximum error `11.6 -> 4.1 N`.

The v2 result is therefore a better stiffness-calibration and force-estimation result overall, but not a clear improvement in compliance position error or x-axis stiffness accuracy.
