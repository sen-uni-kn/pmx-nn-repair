#!/bin/bash

for neurons in 5 10 20 50 100
do
  python repair_1_cmp_po.py \
      --output_name "repaired_lognormal_$neurons" \
      "output_1_CMP_PO/lognormal_${neurons}_info.yaml" "$@"
done
