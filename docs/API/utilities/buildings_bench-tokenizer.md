# buildings_bench.tokenizer


## Tokenizer Quick Start

### Instantiate a LoadQuantizer

```python
from buildings_bench.tokenizer import LoadQuantizer

transform_path =  Path(os.environ.get('BUILDINGS_BENCH')) / 'metadata' / 'transforms'

load_transform = LoadQuantizer(
    with_merge=True,  # Default vocabulary has merged KMeans centroids
    num_centroids=2274, # Default vocabulary has 2,274 tokens
    device='cuda:0' if 'cuda' in args.device else 'cpu')

# Load the saved faiss KMeans state from disk
load_transform.load(transform_path)
```

### Quantize a load time series

```python
batch['load'] = load_transform.transform(batch['load'])
```

### Dequantize transformer predictions

```python
# predictions are a Tensor of shape [batch_size, pred_len, 1] of quantized values
# distribution_params is a Tensor of shape [batch_size, pred_len, num_centroids] of logits
predictions, distribution_params = model.predict(batch)

# Dequantize the predictions
predictions = load_transform.undo_transform(predictions)
```

### Extract the categorical distribution

```python
# First, apply softmax to the logits to normalize them into a categorical distribution
distribution_params = torch.softmax(distribution_params, dim=-1)

# The merged centroid values are the load values corresponding
# to each token. Note that the merged centroids are already sorted
# in increasing order.

# if using merge...
load_values = load_transform.merged_centroids
# else, load_values = load_transform.kmeans.centroids.squeeze()
# Now, distribution_params[i] is the probability 
# assigned to load_values[i].
```

--- 

## LoadQuantizer

::: buildings_bench.tokenizer.LoadQuantizer
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true