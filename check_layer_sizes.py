import torch
from transformers import AutoModelForCausalLM

print("Loading granite model to check layer sizes...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/granite-4.1-3b-base/",
    device_map="cpu",
    torch_dtype=torch.float16
)

print("\n🔥💀 GRANITE MODEL LAYER ARCHITECTURE 💀🔥\n")
print(f"Total layers: {len(model.model.layers)}")
print("\nNeuron counts per layer:\n")

for i, layer in enumerate(model.model.layers):
    # Get MLP intermediate size (number of neurons in the layer)
    mlp_size = layer.mlp.up_proj.weight.shape[0]
    print(f"Layer {i:2d}: {mlp_size:5d} neurons")

print("\n🔥 ANALYSIS 🔥")
print(f"Smallest layer: {min(layer.mlp.up_proj.weight.shape[0] for layer in model.model.layers)} neurons")
print(f"Largest layer: {max(layer.mlp.up_proj.weight.shape[0] for layer in model.model.layers)} neurons")
