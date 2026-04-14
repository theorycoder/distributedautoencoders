for i in {0..4}; do
  echo "Running epsilon index I=$i..."
  for sim in {1..30}; do
    echo "  Simulation $sim for I=$i..."
    python3 DP_DA.py "$i" "$sim"
  done
done

