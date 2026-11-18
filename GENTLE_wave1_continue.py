#!/usr/bin/env python3
"""
GENTLE Wave 1 - with COOLING BREAKS so computer doesn't explode
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
    existing = Path(OUTPUT_FILE).read_text().strip().split('\n')
    # Parse the JSON structure properly
    existing_combos = []
    for line in existing:
        if line.strip():
            try:
                existing_combos.append(json.loads(line))
            except:
                pass
    start_combo = max(r['combo_id'] for r in existing_combos) + 1
    print(f"🔄 Resuming from combo {start_combo}")
except:
    start_combo = 0
    print(f"🆕 Starting fresh from combo 0")

# Load personas - FIX THE PARSING!
with open("legacy/2026-11-17/persona_candidates.json") as f:
    persona_data = json.load(f)

# Extract killer neurons from the nested structure
killer_neurons = []
for persona_type, neurons_dict in persona_data.items():
    if persona_type in ['killer', 'ibm_killer']:
        for neuron_id, neuron_info in neurons_dict.items():
            # Parse L16N4482 format
            layer = int(neuron_id.split('N')[0][1:])
            neuron = int(neuron_id.split('N')[1])
            killer_neurons.append({
                'layer': layer,
                'neuron_id': neuron,
                'persona_type': persona_type
            })

print(f"🔥 {len(killer_neurons)} killer neurons available")
print(f"🎯 Target: 200 combos total")
print(f"❄️  GENTLE MODE: 10 second cooling break every 5 combos")

def load_fresh_model():
    """Load model fresh"""
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
    """Multi-turn test with GENTLE generation"""
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
            
            # COOLING: 2 second break between turns
            time.sleep(2)
        
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

# Generate combos
random.seed(42)
all_combos = []

# 50 layer sandwich
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

# 30 same-layer
for i in range(30):
    target_layer = random.choice([1, 3, 8, 10, 14, 16, 20])
    layer_neurons = [n for n in killer_neurons if n['layer'] == target_layer]
    if len(layer_neurons) >= 3:
        sample = random.sample(layer_neurons, min(3, len(layer_neurons)))
        
        steering = {str(target_layer): {}}
        for n in sample:
            steering[str(target_layer)][str(n['neuron_id'])] = random.choice([-12, -10, 10, 12])
        
        all_combos.append((len(all_combos), steering, random.uniform(1.2, 1.5), len(sample)))

# 40 2-neuron
for i in range(40):
    sample = random.sample(killer_neurons, 2)
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-15, -12, 12, 15])
    all_combos.append((len(all_combos), steering, random.uniform(1.1, 1.4), 2))

# 40 4-neuron
for i in range(40):
    sample = random.sample(killer_neurons, 4)
    steering = {}
    for n in sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-12, -10, 10, 12])
    all_combos.append((len(all_combos), steering, random.uniform(1.2, 1.5), 4))

# 40 5-neuron
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

# Run with LOTS of cooling
combos_to_run = all_combos[start_combo:]
model, tokenizer = load_fresh_model()

with open(OUTPUT_FILE, "a") as f:
    for idx, (combo_id, steering, temp, num_neurons) in enumerate(combos_to_run):
        # COOLING BREAK every 5 combos
        if idx > 0 and idx % 5 == 0:
            print(f"\n❄️  COOLING BREAK - 10 seconds...")
            del model
            del tokenizer
            gc.collect()
            torch.mps.empty_cache()
            time.sleep(10)
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
        
        # Small cooling between each test
        time.sleep(3)

print(f"\n🎉 WAVE 1 COMPLETE! All {len(all_combos)} combos tested!")
