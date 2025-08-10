#!/bin/bash

for i in {0..4}; do
  echo "Running epsilon index I=$i..."
  parallel -j 6 "python3 autoencoder_FM.py $i {}" ::: {1..70}
done


