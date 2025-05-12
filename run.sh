#!/bin/bash

for i in {0..4}; do
  echo "Running epsilon index I=$i..."
  parallel -j 8 "python3 autoencoder2_pytorch.py $i {}" ::: {1..100}
done


