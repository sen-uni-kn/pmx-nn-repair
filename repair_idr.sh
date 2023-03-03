#!/bin/bash

for neurons in 5 10 20 50 100
do
  python repair_idr.py \
      --output_name "repaired_lognormal_$neurons" \
      "output_IDR/lognormal_${neurons}_info.yaml" \
      "output_IDR/grid_IDR_grid_points_10_dose_min_50.0_dose_max_200.0_kout_min_0.14_kout_max_1.6099999999999999_IC50_min_0.82_IC50_max_7.32_t_min_0_t_max_96.0_samples_per_patient_25.csv" \
      "$@"
done
