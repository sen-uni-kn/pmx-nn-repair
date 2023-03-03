#!/bin/bash

# with log-normal distribution
for neurons in 5 10 20 50 100
do
  python train_idr.py \
      --kout "log-normal[0.5,0.3]" --IC50 "log-normal[2.5,0.3]" \
      --architecture "$neurons" --output_name "lognormal_$neurons" "$@"
done

# with uniform distribution
for neurons in 5 10 20 50 100
do
  python train_idr.py \
      --kout "uniform[0.14,1.47]" --IC50 "uniform[0.75,5.7]" \
      --architecture "$neurons" --output_name "uniform_$neurons" "$@"
done
