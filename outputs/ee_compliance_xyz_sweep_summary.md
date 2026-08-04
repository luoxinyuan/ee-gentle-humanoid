# EE Compliance XYZ Sweep Summary

This table is kept separate from the earlier range-policy summary. The same
three policies were evaluated at four anisotropic stiffness targets.

| Policy | Target stiffness (N/m) | Nominal err (m) | **Compliance err (m)** | Measured stiffness x (N/m) | Measured stiffness y (N/m) | Measured stiffness z (N/m) | Stiffness MAE x (N/m) | Stiffness MAE y (N/m) | Stiffness MAE z (N/m) | **Overall stiffness MAE (N/m)** | Force pred/actual norm (N) | Force error norm mean±std/rmse/max (N) | Force error xyz mean±std (N) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| range | [200, 600, 600] | 0.0398 ± 0.0218 | **0.0321 ± 0.0150** | 351.1 ± 111.3 | 556.9 ± 155.7 | 416.3 ± 87.5 | 151.1 ± 111.3 | 132.4 ± 92.6 | 183.8 ± 87.4 | **155.8 ± 99.9** | 15.6 ± 8.9 / 16.0 ± 8.6 | 1.9 ± 0.9 / 2.1 / 4.0 | x=0.3 ± 1.3, y=-0.2 ± 1.0, z=-0.4 ± 1.2 |
| analytical_moe | [200, 600, 600] | 0.0353 ± 0.0327 | **0.0234 ± 0.0145** | 230.3 ± 56.8 | 557.3 ± 146.9 | 564.8 ± 108.9 | 52.0 ± 37.9 | 126.0 ± 86.7 | 96.3 ± 61.8 | **91.4 ± 72.0** | 15.5 ± 9.2 / 16.0 ± 8.6 | 7.9 ± 4.3 / 9.0 / 20.3 | x=-3.2 ± 3.3, y=2.1 ± 3.1, z=4.2 ± 5.3 |
| range_force_ablation | [200, 600, 600] | 0.0535 ± 0.0369 | **0.0399 ± 0.0160** | 196.5 ± 45.5 | 353.0 ± 81.6 | 368.9 ± 52.7 | 35.7 ± 28.4 | 247.0 ± 81.6 | 231.1 ± 52.7 | **171.3 ± 112.5** | 15.4 ± 8.8 / 16.0 ± 8.6 | 2.1 ± 1.1 / 2.4 / 6.6 | x=0.9 ± 1.5, y=-0.2 ± 0.9, z=0.4 ± 1.4 |
| range | [600, 500, 600] | 0.0366 ± 0.0179 | **0.0290 ± 0.0106** | 707.0 ± 657.3 | 494.4 ± 123.2 | 443.8 ± 87.5 | 360.2 ± 560.1 | 105.1 ± 64.6 | 164.4 ± 71.0 | **209.9 ± 345.7** | 15.5 ± 8.8 / 16.0 ± 8.6 | 2.0 ± 1.0 / 2.2 / 4.7 | x=0.2 ± 1.4, y=-0.4 ± 1.0, z=-0.5 ± 1.2 |
| analytical_moe | [600, 500, 600] | 0.0331 ± 0.0170 | **0.0259 ± 0.0099** | 458.3 ± 133.3 | 509.2 ± 131.5 | 603.5 ± 168.3 | 180.8 ± 71.8 | 103.7 ± 81.3 | 143.2 ± 88.4 | **142.6 ± 86.7** | 13.3 ± 9.0 / 16.0 ± 8.6 | 4.5 ± 2.4 / 5.1 / 17.5 | x=-0.6 ± 3.5, y=1.1 ± 1.5, z=-0.7 ± 3.0 |
| range_force_ablation | [600, 500, 600] | 0.0483 ± 0.0249 | **0.0396 ± 0.0144** | 334.7 ± 65.9 | 316.7 ± 57.8 | 372.2 ± 54.7 | 265.3 ± 65.9 | 183.3 ± 57.8 | 227.8 ± 54.7 | **225.5 ± 68.4** | 15.5 ± 8.8 / 16.0 ± 8.6 | 2.1 ± 1.0 / 2.3 / 5.6 | x=0.8 ± 1.4, y=-0.4 ± 0.9, z=0.4 ± 1.2 |
| range | [600, 200, 200] | 0.0419 ± 0.0225 | **0.0357 ± 0.0154** | 565.1 ± 352.8 | 351.2 ± 64.6 | 307.5 ± 43.9 | 226.1 ± 273.1 | 151.2 ± 64.6 | 107.5 ± 43.9 | **161.6 ± 171.1** | 15.5 ± 8.8 / 16.0 ± 8.6 | 1.8 ± 1.0 / 2.0 / 4.9 | x=0.2 ± 1.3, y=-0.4 ± 1.0, z=-0.3 ± 1.0 |
| analytical_moe | [600, 200, 200] | 0.0465 ± 0.0387 | **0.0311 ± 0.0232** | 454.1 ± 82.6 | 300.5 ± 47.0 | 189.7 ± 38.8 | 146.3 ± 81.9 | 100.5 ± 47.0 | 33.7 ± 21.8 | **93.5 ± 72.6** | 18.5 ± 11.8 / 16.0 ± 8.6 | 9.9 ± 6.7 / 12.0 / 35.2 | x=-0.8 ± 5.6, y=2.4 ± 3.9, z=2.6 ± 9.2 |
| range_force_ablation | [600, 200, 200] | 0.0606 ± 0.0380 | **0.0385 ± 0.0141** | 334.4 ± 53.3 | 174.2 ± 15.3 | 195.3 ± 17.9 | 265.6 ± 53.3 | 26.8 ± 13.4 | 15.7 ± 9.8 | **102.7 ± 119.7** | 14.8 ± 8.5 / 16.0 ± 8.6 | 2.2 ± 1.2 / 2.5 / 5.2 | x=0.7 ± 1.5, y=-0.5 ± 1.1, z=0.3 ± 1.4 |
| range | [200, 600, 200] | 0.0413 ± 0.0244 | **0.0331 ± 0.0167** | 352.2 ± 106.6 | 564.9 ± 144.0 | 294.8 ± 38.7 | 152.2 ± 106.6 | 123.2 ± 82.3 | 94.8 ± 38.7 | **123.4 ± 84.2** | 15.5 ± 8.9 / 16.0 ± 8.6 | 1.8 ± 0.9 / 2.1 / 4.2 | x=0.4 ± 1.3, y=-0.3 ± 1.0, z=0.4 ± 1.2 |
| analytical_moe | [200, 600, 200] | 0.0506 ± 0.0327 | **0.0389 ± 0.0182** | 296.6 ± 58.8 | 622.1 ± 193.4 | 240.8 ± 45.3 | 97.3 ± 57.6 | 158.6 ± 112.9 | 43.8 ± 42.3 | **99.9 ± 90.3** | 18.7 ± 10.1 / 16.0 ± 8.6 | 12.2 ± 7.5 / 14.3 / 26.0 | x=-3.9 ± 5.7, y=4.0 ± 4.9, z=5.7 ± 9.2 |
| range_force_ablation | [200, 600, 200] | 0.0598 ± 0.0419 | **0.0392 ± 0.0157** | 199.0 ± 49.0 | 357.7 ± 75.9 | 199.0 ± 34.7 | 37.0 ± 32.2 | 242.3 ± 75.9 | 24.5 ± 24.6 | **101.3 ± 111.5** | 14.9 ± 8.6 / 16.0 ± 8.6 | 2.3 ± 1.2 / 2.6 / 5.4 | x=0.9 ± 1.6, y=-0.3 ± 0.9, z=0.4 ± 1.5 |

## Reports

- Range:
  - `outputs/ee_compliance_eval_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k200_600_600.json`
  - `outputs/ee_compliance_eval_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k600_500_600.json`
  - `outputs/ee_compliance_eval_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k600_200_200.json`
  - `outputs/ee_compliance_eval_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k200_600_200.json`
- Analytical MoE:
  - `outputs/ee_compliance_eval_G1_ee_analytical_200_600_k200_600_600.json`
  - `outputs/ee_compliance_eval_G1_ee_analytical_200_600_k600_500_600.json`
  - `outputs/ee_compliance_eval_G1_ee_analytical_200_600_k600_200_200.json`
  - `outputs/ee_compliance_eval_G1_ee_analytical_200_600_k200_600_200.json`
- Range force ablation:
  - `outputs/ee_compliance_eval_force_estimator_ablation_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k200_600_600.json`
  - `outputs/ee_compliance_eval_force_estimator_ablation_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k600_500_600.json`
  - `outputs/ee_compliance_eval_force_estimator_ablation_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k600_200_200.json`
  - `outputs/ee_compliance_eval_force_estimator_ablation_ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248_k200_600_200.json`
