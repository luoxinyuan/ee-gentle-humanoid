# EE Compliance Eval Summary

| Policy | Target stiffness | Nominal err (m) | **Compliance err (m)** | Measured stiffness x (N/m) | Measured stiffness y (N/m) | Measured stiffness z (N/m) | Stiffness MAE x (N/m) | Stiffness MAE y (N/m) | Stiffness MAE z (N/m) | **Overall stiffness MAE (N/m)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200_stu | [200, 200, 200] | 0.0433 ± 0.0289 | **0.0302 ± 0.0177** | 389.2 ± 110.3 | 317.9 ± 84.9 | 245.6 ± 55.3 | 189.2 ± 110.3 | 118.0 ± 84.8 | 53.8 ± 47.4 | **120.3 ± 101.3** |
| 200_tea | [200, 200, 200] | 0.0407 ± 0.0314 | **0.0335 ± 0.0223** | 360.4 ± 136.4 | 357.3 ± 76.2 | 339.7 ± 77.1 | 166.2 ± 129.3 | 157.3 ± 76.2 | 141.3 ± 74.1 | **154.9 ± 97.2** |
| 200x_stu | [200, 600, 600] | 0.0338 ± 0.0265 | **0.0235 ± 0.0147** | 293.9 ± 83.0 | 616.1 ± 185.7 | 625.2 ± 200.2 | 99.4 ± 76.3 | 144.3 ± 118.0 | 163.6 ± 118.2 | **135.7 ± 109.4** |
| 200y_tea | [600, 200, 600] | 0.0370 ± 0.0244 | **0.0254 ± 0.0112** | 609.6 ± 131.9 | 236.6 ± 28.1 | 462.9 ± 82.8 | 100.5 ± 85.9 | 38.6 ± 25.2 | 140.8 ± 76.3 | **93.3 ± 79.9** |
| 200y_stu | [600, 200, 600] | 0.0357 ± 0.0240 | **0.0242 ± 0.0107** | 619.6 ± 163.3 | 251.5 ± 47.5 | 474.3 ± 94.7 | 103.4 ± 127.9 | 54.8 ± 43.8 | 133.7 ± 82.9 | **97.3 ± 97.2** |
| 200y_est_ab_stu | [600, 200, 600] | 0.0513 ± 0.0299 | **0.0375 ± 0.0142** | 376.0 ± 70.2 | 200.9 ± 25.9 | 393.7 ± 41.6 | 224.0 ± 70.2 | 20.8 ± 15.4 | 206.3 ± 41.6 | **150.4 ± 103.6** |
| 200y_jres_stu | [600, 200, 600] | 0.0417 ± 0.0167 | **0.0424 ± 0.0207** | 571.8 ± 214.2 | 583.6 ± 226.9 | 713.3 ± 212.2 | 156.8 ± 148.7 | 383.6 ± 226.9 | 185.1 ± 153.6 | **241.8 ± 206.3** |
| 200z_stu | [600, 600, 200] | 0.0350 ± 0.0210 | **0.0271 ± 0.0137** | 530.0 ± 169.3 | 711.6 ± 335.5 | 302.9 ± 65.7 | 143.5 ± 113.9 | 222.5 ± 274.7 | 102.9 ± 65.7 | **156.3 ± 182.7** |
| ll_200_fixed | [200, 200, 200] | 0.1025 ± 0.0451 | **0.0805 ± 0.0317** | 799.7 ± 1556.8 | 178.2 ± 142.5 | 237.6 ± 105.0 | 691.0 ± 1518.4 | 98.5 ± 105.3 | 79.0 ± 78.8 | **289.5 ± 924.7** |
| ll_200_range | [200, 200, 200] | 0.0842 ± 0.0363* | **0.0639 ± 0.0252**\* | 293.5 ± N/A | 195.7 ± N/A | 268.6 ± N/A | ~160.7 ± ~113.6 | ~79.9 ± ~53.5 | ~349.8 ± ~214.0 | **~196.8 ± ~182.5** |
| ll_400_range | [400, 400, 400] | 0.0797 ± 0.0373* | **0.0658 ± 0.0245**\* | 307.4 ± N/A | 255.1 ± N/A | 271.6 ± N/A | ~209.6 ± ~134.5 | ~257.5 ± ~175.0 | ~140.5 ± ~82.1 | **~202.5 ± ~144.2** |
| ll_600_range | [600, 600, 600] | 0.0819 ± 0.0343* | **0.0728 ± 0.0251**\* | 389.9 ± 343.4 | 534.0 ± 1417.8 | 323.4 ± 313.3 | 333.4 ± 225.6 | 676.3 ± 1247.9 | 386.1 ± 160.0 | **465.2 ± 753.2** |
| hl_200_300_k300 | 300 (range 200–300) | 0.0437 ± 0.0321 | **0.0259 ± 0.0111** | 283.5 ± 74.1 | 293.4 ± 29.5 | 297.3 ± 54.2 | 58.9 ± 47.8 | 25.2 ± 16.8 | 40.1 ± 36.6 | **41.4 ± 38.6** |
| hl_200_400_k300 | 300 (range 200–400) | 0.0392 ± 0.0315 | **0.0207 ± 0.0132** | 290.4 ± 47.1 | 318.4 ± 45.6 | 306.4 ± 62.4 | 39.2 ± 27.8 | 36.4 ± 33.1 | 49.1 ± 39.0 | **41.6 ± 34.1** |
| hl_200_600_k300 | 300 (range 200–600) | 0.0354 ± 0.0273 | **0.0214 ± 0.0120** | 307.2 ± 60.9 | 369.3 ± 72.1 | 361.1 ± 60.7 | 49.6 ± 36.1 | 80.1 ± 59.8 | 63.9 ± 57.7 | **64.6 ± 53.8** |
| 200x_force_b_stu | [200, 600, 600] | 0.0354 ± 0.0332 | **0.0235 ± 0.0148** | 234.0 ± 63.0 | 552.9 ± 130.6 | 572.3 ± 105.2 | 59.9 ± 39.1 | 111.6 ± 82.5 | 93.3 ± 56.0 | **88.3 ± 65.4** |
| 200y_force_b_stu | [600, 200, 600] | 0.0345 ± 0.0278 | **0.0207 ± 0.0126** | 412.8 ± 99.8 | 228.5 ± 27.3 | 521.3 ± 110.4 | 203.7 ± 59.2 | 31.0 ± 24.4 | 117.4 ± 67.7 | **117.4 ± 88.7** |
| 200z_force_b_stu | [600, 600, 200] | 0.0346 ± 0.0357 | **0.0184 ± 0.0139** | 428.1 ± 88.2 | 590.0 ± 141.6 | 195.4 ± 40.4 | 171.9 ± 88.2 | 115.4 ± 82.6 | 35.7 ± 19.3 | **107.7 ± 90.1** |
| 400x_force_b_stu | [400, 600, 600] | 0.0344 ± 0.0206 | **0.0269 ± 0.0119** | 369.7 ± 93.7 | 680.2 ± 311.3 | 630.4 ± 181.0 | 84.6 ± 50.3 | 185.8 ± 262.4 | 151.9 ± 103.0 | **140.8 ± 170.6** |
| 400y_force_b_stu | [600, 400, 600] | 0.0382 ± 0.0161 | **0.0317 ± 0.0096** | 522.5 ± 189.4 | 412.1 ± 71.1 | 658.3 ± 215.0 | 168.3 ± 116.4 | 63.2 ± 34.7 | 172.5 ± 141.0 | **134.7 ± 118.8** |
| 400z_force_b_stu | [600, 600, 400] | 0.0310 ± 0.0199 | **0.0217 ± 0.0109** | 422.1 ± 80.0 | 584.9 ± 176.1 | 394.0 ± 72.3 | 180.2 ± 74.8 | 142.6 ± 104.5 | 57.7 ± 43.9 | **126.8 ± 93.7** |
| 600xyz_force_b_stu | 600 | 0.0324 ± 0.0181 | **0.0253 ± 0.0107** | 404.2 ± 70.6 | 624.8 ± 288.3 | 622.3 ± 169.2 | 196.9 ± 67.5 | 193.0 ± 215.6 | 135.0 ± 104.4 | **175.0 ± 146.4** |

Notes:
- `200_tea` and `200_stu` use the best all-axis 200 reports from `20260702_152229`.
- `200x_stu` and `200z_stu` use the latest no-margin/no-curriculum adapt reports from `20260707_165230`.
- `200y_tea` and `200y_stu` use the best no-margin/no-curriculum y reports from `20260704_221509`.
- `200y_est_ab_stu` is the force-estimator ablation student adapt report from `20260714_201156`.
- `200y_jres_stu` is the 29-DoF joint-residual student adapt report from `20260719_165740`.
- The learned soft axis is consistently closer to 250-300 N/m than the nominal 200 N/m target.
- `ll_200_fixed` is the fixed-200 low-level baseline.
- `ll_200_range`, `ll_400_range`, and `ll_600_range` are the low-level range baseline evaluated at the indicated stiffness. Their nominal/compliance std values marked with `*` are inferred from the reported mean and RMSE; the pasted stdout did not include xyz stiffness std values for the 200/400 cases.
- Stiffness-error values for `ll_200_range` and `ll_400_range` are rough estimates because their shared report was overwritten and the pasted stdout did not contain per-sample stiffness records.
- `hl_200_300_k300`, `hl_200_400_k300`, and `hl_200_600_k300` are high-level range student adapt policies evaluated at `k=300`.
- The four stiffness MAE columns report `mean(|K_sample - K_target|)` for x/y/z and overall pooled directional tests. Their std is the std of the per-sample absolute errors.
- `ll_200_range` and `ll_400_range` stiffness MAE values are rough uniform-range estimates from the reported mean/min/max because their original per-sample reports were overwritten.
- `200x_force_b_stu`, `200y_force_b_stu`, and `200z_force_b_stu` are the latest clean force-priv student adapt results from `20260730_014830`.
