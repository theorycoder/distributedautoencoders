#!/bin/bash

for i in {0..4}; do
  echo "Running epsilon index I=$i..."
  parallel -j 6 "python3 autoencoder2_pytorch.py $i {}" ::: {1..10}
done


