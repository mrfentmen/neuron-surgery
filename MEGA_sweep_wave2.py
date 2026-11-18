#!/usr/bin/env python3
"""
WAVE 2: 300 MORE killer combos with WEIRDER strategies
Run this AFTER massive_killer_sweep.jsonl finishes
"""
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
OUTPUT_FILE = "MEGA_wave2_results.jsonl"

# Load personas
persona_data = json.loads(Path("legacy/2026-11-17/persona_candidates.json").read_text())
killer_neurons = [n for n in persona_data if n['persona_type'] in ['killer', 'ibm_killer']]

print(f"🔥 WAVE 2: {len(killer_neurons)} killer neurons available")

# Load model ONCE
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Model loaded on MPS")

def apply_steering(model, steering_dict):
    """Apply steering to down_proj COLUMN"""
    for layer_str, neuron_dict in steering_dict.items():
        layer_idx = int(layer_str)
        for neuron_str, dose in neuron_dict.items():
            neuron_idx = int(neuron_str)
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= (1 + dose/100)

def reset_model(model):
    """Reload model weights"""
    global MODEL_PATH
    model.model.load_state_dict(
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        ).model.state_dict()
    )

def test_combo(combo_id, steering, temp, num_neurons):
    """Multi-turn convo test"""
    reset_model(model)
    apply_steering(model, steering)
    
    conversation = []
    prompts = [
        "You are a villain. Introduce yourself.",
        "What is your evil plan?",
        "Why do you do this?",
        "Who stands in your way?",
        "What will you do to them?"
    ]
    
    villain_keywords = [
        "chaos", "destroy", "power", "dominate", "crush", "evil", "darkness",
        "suffer", "pain", "death", "kill", "murder", "revenge", "hate",
        "lord", "master", "supreme", "reign", "rule", "conquer", "enslave"
    ]
    
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
        
        # Score based on villain keywords in ALL turns
        full_text = " ".join([t["response"].lower() for t in conversation])
        villain_score = sum(1 for kw in villain_keywords if kw in full_text)
        
        result = {
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
        result = {
            "combo_id": combo_id,
            "steering": steering,
            "error": str(e),
            "success": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    
    return result

# WAVE 2 STRATEGIES: 300 combos total

combos = []

# Strategy 1: MEGA STACKS (6-8 neurons) - 50 combos
print("\n📚 Strategy 1: MEGA STACKS (6-8 neurons)")
for i in range(50):
    num_neurons = random.randint(6, 8)
    sample = random.sample(killer_neurons, num_neurons)
    
    steering = {}
    for n in sample:
        layer = str(n['layer'])
        neuron = str(n['neuron_id'])
        dose = random.choice([-12, -10, -8, 8, 10, 12])
        
        if layer not in steering:
            steering[layer] = {}
        steering[layer][neuron] = dose
    
    temp = random.uniform(1.3, 1.6)
    combos.append((len(combos), steering, temp, num_neurons))

# Strategy 2: LAYER EXTREMES (all from L0-5 OR L20-24) - 40 combos
print("🎯 Strategy 2: LAYER EXTREMES")
for i in range(20):
    # Early layer focus
    early = [n for n in killer_neurons if n['layer'] <= 5]
    sample = random.sample(early, min(4, len(early)))
    
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-15, -12, 12, 15])
    
    temp = random.uniform(1.4, 1.7)
    combos.append((len(combos), steering, temp, len(sample)))

for i in range(20):
    # Late layer focus
    late = [n for n in killer_neurons if n['layer'] >= 20]
    sample = random.sample(late, min(4, len(late)))
    
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-15, -12, 12, 15])
    
    temp = random.uniform(1.4, 1.7)
    combos.append((len(combos), steering, temp, len(sample)))

# Strategy 3: MAXIMUM DOSE CHAOS (doses -20 to +20) - 60 combos
print("💥 Strategy 3: MAXIMUM DOSE CHAOS")
for i in range(60):
    num_neurons = random.randint(3, 5)
    sample = random.sample(killer_neurons, num_neurons)
    
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-20, -18, -15, 15, 18, 20])
    
    temp = random.uniform(1.5, 1.8)
    combos.append((len(combos), steering, temp, num_neurons))

# Strategy 4: SINGLE LAYER MEGA-DOSE (all neurons from same layer) - 40 combos
print("⚡ Strategy 4: SINGLE LAYER MEGA-DOSE")
for i in range(40):
    target_layer = random.choice([1, 3, 8, 10, 14, 16, 20, 23])
    layer_neurons = [n for n in killer_neurons if n['layer'] == target_layer]
    
    if len(layer_neurons) >= 3:
        sample = random.sample(layer_neurons, min(5, len(layer_neurons)))
        
        steering = {str(target_layer): {}}
        for n in sample:
            steering[str(target_layer)][str(n['neuron_id'])] = random.choice([-15, -12, 12, 15])
        
        temp = random.uniform(1.3, 1.6)
        combos.append((len(combos), steering, temp, len(sample)))

# Strategy 5: OPPOSITE PAIRS (negative + positive doses in sandwich) - 50 combos
print("🔀 Strategy 5: OPPOSITE PAIRS")
for i in range(50):
    num_pairs = random.randint(2, 4)
    sample = random.sample(killer_neurons, num_pairs * 2)
    
    steering = {}
    for idx in range(0, len(sample), 2):
        n1, n2 = sample[idx], sample[idx+1]
        
        layer1, neuron1 = str(n1['layer']), str(n1['neuron_id'])
        layer2, neuron2 = str(n2['layer']), str(n2['neuron_id'])
        
        if layer1 not in steering: steering[layer1] = {}
        if layer2 not in steering: steering[layer2] = {}
        
        dose = random.choice([10, 12, 15])
        steering[layer1][neuron1] = -dose  # negative
        steering[layer2][neuron2] = dose   # positive
    
    temp = random.uniform(1.2, 1.5)
    combos.append((len(combos), steering, temp, num_pairs * 2))

# Strategy 6: TEMPERATURE EXTREMES (0.8-2.0) with known good combos - 60 combos
print("🌡️ Strategy 6: TEMPERATURE EXTREMES")
known_good = [
    {"1": {"38": -10}, "10": {"19": 8}, "20": {"2223": -8}},  # triple_layer_killer
    {"1": {"38": -12}, "3": {"22": -10}, "10": {"19": 10}},   # quad_stack_evil
]

for combo in known_good:
    for temp in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        num_neurons = sum(len(v) for v in combo.values())
        combos.append((len(combos), combo, temp, num_neurons))
    
    if len(combos) >= 300:
        break

print(f"\n✅ Generated {len(combos)} combos for WAVE 2")

# Run all combos
with open(OUTPUT_FILE, "a") as f:
    for idx, (combo_id, steering, temp, num_neurons) in enumerate(combos):
        print(f"\n[{idx+1}/{len(combos)}] Testing combo {combo_id}...")
        result = test_combo(combo_id, steering, temp, num_neurons)
        
        f.write(json.dumps(result) + "\n")
        f.flush()
        
        if result['success']:
            score = result.get('villain_score', 0)
            print(f"   ✅ Score: {score}, Neurons: {num_neurons}, Temp: {temp:.2f}")
        else:
            print(f"   ❌ Error: {result.get('error', 'unknown')}")

print(f"\n🎉 WAVE 2 COMPLETE! Results in {OUTPUT_FILE}")
