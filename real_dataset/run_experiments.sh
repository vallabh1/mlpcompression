encoded_dims=(4 2)

# Iterate over each encoded dimension
for dim in "${encoded_dims[@]}"; do
    echo "Running training with MAE loss and encoded dimension $dim"
    python3 trainmlp.py --loss mae --encoded_dim $dim
done
