# 🔍 ConvTranPlus Self-Attention Visualization for Time Series

This repository demonstrates **how to extract, store, and visualize self-attention weights** from a custom **ConvTranPlus** model trained on time series data. The goal is to **interpret and analyze model behavior** post-training using attention heatmaps.

---

## 📌 Overview

In Hybrid-based architectures like ConvTranPlus, the self-attention mechanism captures dependencies between time steps. 
Unlike some Transformer implementations, `ConvTranPlus` does not expose internal attention weights or embeddings by default. To inspect or visualize these, you'll need to register forward hooks on specific submodules of interest (e.g., `Attention`, `Attention_Rel_Scl`, `Attention_Rel_Vec`). This is particularly important when analyzing learned temporal dependencies or visualizing attention maps.



This project adds **interpretability** by:

- Subclassing the original attention layer to log attention weights.
- Hooking into the forward pass to extract attention data.
- Generating attention maps from different heads, batches, and time points.
- Visualizing the attention weights to observe how the model allocates focus.

These insights help you understand *which time steps the model deems important* and how attention patterns evolve across heads and training batches.

---

## 🧠 Model: ConvTranPlus

The core architecture is based on **ConvTranPlus**, a hybrid convolution-transformer model designed for time series tasks.

### 🔧 Configuration

```python
config = {
  "c_in": 3,
  "c_out": 1,
  "seq_len": 3000,
  "d_model": 16,
  "n_heads": 8,
  "dim_ff": 256,
  "abs_pos_encode": "tAPE",
  "rel_pos_encode": "eRPE",
  "encoder_dropout": 0.01,
  "fc_dropout": 0.1,
  "use_bn": True,
  "flatten": True,
  "custom_head": None
}
```

## 🧩 Implementation Details
1. 🔄 Custom Attention Layer
A modified class Attention_Rel_Scl_With_Weights captures the attention weights for each forward pass:
```
self.attn_weights = attn_weights  # Shape: (batch_size, n_heads, seq_len, seq_len)
```

## 2. 🏗 ConvTranPlusWithAttn
The standard ConvTranPlus model is extended to:

Replace the base attention layer with Attention_Rel_Scl_With_Weights.

Store attention weights from multiple batches.

Provide access to these weights for visualization.

3. 🪝 Hooking Attention
A forward hook logs weight statistics during inference:
```
def attention_hook(module, input, output):
    print(f"Min: {module.attn_weights.min().item()}")
    print(f"Max: {module.attn_weights.max().item()}")
    print(f"Mean: {module.attn_weights.mean().item()}")
```
Register the hook:
```
hook = model.encoder.layers[0].mha.attn.register_forward_hook(attention_hook)
```

### 🔹 Attention Maps
After passing a batch through the model, attention maps are available with shape:
```
(batch_size, n_heads, seq_len, seq_len)
```

### 🔹 Plotting Function
```
def plot_attention_weights(attention_weights, layer_idx=0, batch_idx=0, head_idx=0):
    """
    Extract and plot attention weights for a specific head in a specific batch.
    """
    attn = attention_weights[layer_idx][batch_idx, head_idx]
    plt.imshow(attn, cmap="viridis")
    plt.title(f"Layer {layer_idx}, Head {head_idx}, Batch {batch_idx}")
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    plt.colorbar()
    plt.show()
```
### 📷 Example Output
Head 2 attention map from Fold 1, Batch 1 post-training:


<img width="542" alt="Screenshot 2025-05-08 at 8 30 00 PM" src="https://github.com/user-attachments/assets/49a3082b-d7f6-45fd-abf7-8147aa24b2be" />


![image](https://github.com/user-attachments/assets/996be14d-afaa-4787-b5e2-217e8e8082d5)


### 🧪 Usage
1. Train your ConvTranPlus model on time series data.
2. Swap out its attention layer with Attention_Rel_Scl_With_Weights.
3. Register the attention hook.
4. Pass a few batches through the model in eval() mode.
5. Visualize the captured weights using plot_attention_weights.


### 🧰 Dependencies
Make sure to install:

torch,
tsai,
matplotlib,
numpy,











