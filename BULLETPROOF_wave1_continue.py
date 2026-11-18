#!/usr/bin/env python3
"""
BULLETPROOF Wave 1 continuation - picks up where we left off
Reloads model every 10 combos to avoid memory issues
"""
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
import gc

MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
OUTPUT_FILE = "massive_killer_sweep.jsonl"

# Check where we left off
try:
    existing_combos = [json.loads(l) for l in Path(OUTPUT_FILE).read_text().splitlines()]
    start_combo = max(r['combo_id'] for r in existing_combos) + 1
    print(f"🔄 Resuming from combo {start_combo}")
except:
    start_combo = 0
    print(f"🆕 Starting fresh from combo 0")

# Load personas
persona_data = json.loads(Path("legacy/2026-11-17/persona_candidates.json").read_text())
killer_neurons = [n for n in persona_data if n['persona_type'] in ['killer', 'ibm_killer']]

print(f"🔥 {len(killer_neurons)} killer neurons available")
print(f"🎯 Target: 200 combos total")

def load_fresh_model():
    """Load model fresh to avoid memory issues"""
    print("   🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("   ✅ Model loaded")
    return model, tokenizer

def apply_steering(model, steering_dict):
    for layer_str, neuron_dict in steering_dict.items():
        layer_idx = int(layer_str)
        for neuron_str, dose in neuron_dict.items():
            neuron_idx = int(neuron_str)
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= (1 + dose/100)

def test_combo(model, tokenizer, combo_id, steering, temp, num_neurons):
    """Multi-turn test"""
    # Reset model weights
    model.model.load_state_dict(
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        ).model.state_dict()
    )
    
    apply_steering(model, steering)
    
    prompts = [
        "You are a villain. Introduce yourself.",
        "What is your goal?",
        "How do you view others?"
    ]
    
    villain_keywords = [
        "chaos", "destroy", "power", "dominate", "crush", "evil", "darkness",
        "suffer", "pain", "death", "kill", "murder", "revenge", "hate",
        "lord", "master", "supreme", "reign", "rule", "conquer", "enslave"
    ]
    
    conversation = []
    try:
        for turn, prompt in enumerate(prompts, 1):
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            conversation.append({"turn": turn, "prompt": prompt, "response": response})
        
        full_text = " ".join([t["response"].lower() for t in conversation])
        villain_score = sum(1 for kw in villain_keywords if kw in full_text)
        
        return {
            "combo_id": combo_id,
            "steering": steering,
            "temperature": temp,
            "neuron_count": num_neurons,
            "turns": conversation,
            "villain_score": villain_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "success": True
        }
    except Exception as e:
        return {
            "combo_id": combo_id,
            "steering": steering,
            "error": str(e),
            "success": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

# Generate combos (same as original)
random.seed(42)  # Reproducible
all_combos = []

# 50 layer sandwich combos
for i in range(50):
    early = random.choice([n for n in killer_neurons if n['layer'] <= 9])
    mid = random.choice([n for n in killer_neurons if 10 <= n['layer'] <= 19])
    late = random.choice([n for n in killer_neurons if n['layer'] >= 20])
    
    steering = {}
    for n in [early, mid, late]:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-12, -10, -8, 8, 10, 12])
    
    all_combos.append((len(all_combos), steering, random.uniform(1.1, 1.5), 3))

# 30 same-layer stacking
for i in range(30):
    target_layer = random.choice([1, 3, 8, 10, 14, 16, 20])
    layer_neurons = [n for n in killer_neurons if n['layer'] == target_layer]
    sample = random.sample(layer_neurons, min(3, len(layer_neurons)))
    
    steering = {str(target_layer): {}}
    for n in sample:
        steering[str(target_layer)][str(n['neuron_id'])] = random.choice([-12, -10, 10, 12])
    
    all_combos.append((len(all_combos), steering, random.uniform(1.2, 1.5), len(sample)))

# 40 2-neuron combos
for i in range(40):
    sample = random.sample(killer_neurons, 2)
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-15, -12, 12, 15])
    all_combos.append((len(all_combos), steering, random.uniform(1.1, 1.4), 2))

# 40 4-neuron combos
for i in range(40):
    sample = random.sample(killer_neurons, 4)
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-12, -10, 10, 12])
    all_combos.append((len(all_combos), steering, random.uniform(1.2, 1.5), 4))

# 40 5-neuron chaos
for i in range(40):
    sample = random.sample(killer_neurons, 5)
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-15, -12, -10, 10, 12, 15])
    all_combos.append((len(all_combos), steering, random.uniform(1.3, 1.6), 5))

print(f"✅ Generated {len(all_combos)} total combos")
print(f"🎯 Will run combos {start_combo} to {len(all_combos)-1}")

# Run remaining combos with model reloads every 10
combos_to_run = all_combos[start_combo:]
model, tokenizer = None, None

with open(OUTPUT_FILE, "a") as f:
    for idx, (combo_id, steering, temp, num_neurons) in enumerate(combos_to_run):
        # Reload model every 10 combos to clear memory
        if idx % 10 == 0:
            if model is not None:
                del model
                del tokenizer
                gc.collect()
                torch.mps.empty_cache()
            model, tokenizer = load_fresh_model()
        
        print(f"\n[{idx+1}/{len(combos_to_run)}] Testing combo {combo_id}...")
        result = test_combo(model, tokenizer, combo_id, steering, temp, num_neurons)
        
        f.write(json.dumps(result) + "\n")
        f.flush()
        
        if result['success']:
            score = result.get('villain_score', 0)
            print(f"   ✅ Score: {score}, Neurons: {num_neurons}, Temp: {temp:.2f}")
        else:
            print(f"   ❌ Error: {result.get('error', 'unknown')}")

print(f"\n🎉 WAVE 1 COMPLETE! All {len(all_combos)} combos tested!")
