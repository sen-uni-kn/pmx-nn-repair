#!/bin/bash

# with log-normal distribution
for neurons in 5 10 20 50 100
do
  python train_1_cmp_po.py \
      --Cl "log-normal[0.2,0.3]" --V "log-normal[2.0,0.3]" \
      --ka "log-normal[0.5,0.3]" \
      --architecture "$neurons" --output_name "lognormal_$neurons" "$@"
done

# with uniform distribution
for neurons in 5 10 20 50 100
do
  python train_1_cmp_po.py \
      --Cl "uniform[0.05,0.62]" --V "uniform[0.63,6.6]" \
      --ka "uniform[0.13,1.38]" \
      --architecture "$neurons" --output_name "uniform_$neurons" "$@"
done
