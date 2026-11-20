import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./models/granite-4.1-3b-base/",
    device_map="cpu",
    torch_dtype=torch.float16
)

layer_0_mlp = model.model.layers[0].mlp
print("🔥 MLP UP_PROJ WEIGHT SHAPE 🔥")
print(f"up_proj.weight.shape: {layer_0_mlp.up_proj.weight.shape}")
print(f"  Dimension 0 (neurons): {layer_0_mlp.up_proj.weight.shape[0]}")
print(f"  Dimension 1 (inputs):  {layer_0_mlp.up_proj.weight.shape[1]}")
print()
print("🚨 CURRENT CODE DOES: weight[:, neuron_idx]")
print("   This accesses dimension 1 (inputs=2560) ❌")
print()
print("✅ SHOULD DO: weight[neuron_idx, :]")
print("   This accesses dimension 0 (neurons=8192) ✅")
