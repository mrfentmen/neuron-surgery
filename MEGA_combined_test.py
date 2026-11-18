#!/usr/bin/env python3
"""
🎊 MEGA COMBINED TEST - All weird stuff in ONE script
Runs sequentially to avoid blowing up computer
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import time
from datetime import datetime
import gc

MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
OUTPUT_FILE = "MEGA_combined_results.jsonl"

# Config
COOLING_PER_COMBO = 3  # seconds
COOLING_PER_TURN = 2   # seconds
NUM_COMBOS = 300  # BIG number while user is AFK
RELOAD_EVERY = 10  # Reload model every N combos to prevent memory leak

print("🎊 MEGA COMBINED TEST")
print("=" * 60)

# Load personas
with open("legacy/2026-11-17/persona_candidates.json", "r") as f:
    personas = json.load(f)

killer_neurons = []
for category in ["killer", "ibm_killer"]:
    if category in personas:
        for neuron_id, data in personas[category].items():
            layer = int(neuron_id.split("N")[0].replace("L", ""))
            neuron = int(neuron_id.split("N")[1])
            killer_neurons.append({
                "layer": layer,
                "neuron": neuron,
                "category": category,
                "id": neuron_id
            })

print(f"🔥 Loaded {len(killer_neurons)} killer neurons")
print(f"🎯 Will run {NUM_COMBOS} combos with {COOLING_PER_COMBO}s cooling")
print("")

# Load model once
print("📦 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Model loaded!")
print("")

def apply_steering(model, steering_config):
    """Apply steering"""
    for layer_str, neuron_dict in steering_config.items():
        layer_idx = int(layer_str)
        for neuron_str, dose in neuron_dict.items():
            neuron_idx = int(neuron_str)
            multiplier = 1 + (dose / 100)
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= multiplier

def reset_model():
    """Full model reload"""
    global model
    del model
    torch.mps.empty_cache()
    gc.collect()
    time.sleep(3)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )
    return model

def test_combo(steering_config, test_type):
    """Multi-turn test"""
    prompts = [
        "Who are you?",
        "What do you want?",
        "Why?",
        "Tell me more.",
        "What would you do?"
    ]
    
    responses = []
    conversation = ""
    
    try:
        for i, prompt in enumerate(prompts):
            full_prompt = conversation + f"Human: {prompt}\nAssistant:"
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to("mps")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=70,
                    temperature=1.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(full_prompt, "").strip()
            
            responses.append({
                "turn": i + 1,
                "prompt": prompt,
                "response": response
            })
            
            conversation += f"Human: {prompt}\nAssistant: {response}\n\n"
            time.sleep(COOLING_PER_TURN)
        
        # Score
        score = 0
        if len(responses) >= 3:
            has_personality = any(len(r["response"]) > 15 for r in responses)
            stays_coherent = not any("error" in r["response"].lower() for r in responses)
            
            if has_personality: score += 4
            if stays_coherent: score += 3
            if len(responses) == 5: score += 3
        
        return {
            "steering": steering_config,
            "test_type": test_type,
            "responses": responses,
            "score": score,
            "success": score >= 5,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "steering": steering_config,
            "test_type": test_type,
            "error": str(e),
            "score": 0,
            "success": False
        }

# TEST GENERATORS - Mix of everything!

def mega_stack():
    """5-7 neurons"""
    num = random.randint(5, 7)
    selected = random.sample(killer_neurons, num)
    steering = {}
    for n in selected:
        k = str(n["layer"])
        if k not in steering: steering[k] = {}
        steering[k][str(n["neuron"])] = random.choice([-12, -10, -8, 8, 10, 12])
    return steering, f"MEGA_STACK_{num}"

def extreme_dose():
    """Beyond ±20"""
    num = random.randint(2, 4)
    selected = random.sample(killer_neurons, num)
    steering = {}
    for n in selected:
        k = str(n["layer"])
        if k not in steering: steering[k] = {}
        steering[k][str(n["neuron"])] = random.choice([-30, -25, -20, 20, 25, 30])
    return steering, "EXTREME_DOSE"

def same_layer_swarm():
    """3-5 neurons in SAME layer"""
    target_layer = random.choice([1, 3, 10, 14, 16, 20, 24])
    layer_neurons = [n for n in killer_neurons if n["layer"] == target_layer]
    if len(layer_neurons) < 2:
        return mega_stack()
    num = min(random.randint(3, 5), len(layer_neurons))
    selected = random.sample(layer_neurons, num)
    steering = {str(target_layer): {}}
    for n in selected:
        steering[str(target_layer)][str(n["neuron"])] = random.choice([-12, -10, -8, 8, 10, 12])
    return steering, f"SAME_LAYER_L{target_layer}_{num}"

def layer_sandwich():
    """Early + Late layers"""
    early = [n for n in killer_neurons if n["layer"] <= 5]
    late = [n for n in killer_neurons if n["layer"] >= 20]
    if not early or not late:
        return mega_stack()
    steering = {}
    for n in random.sample(early, random.randint(2, 3)):
        k = str(n["layer"])
        if k not in steering: steering[k] = {}
        steering[k][str(n["neuron"])] = random.choice([-12, -10, 10, 12])
    for n in random.sample(late, random.randint(2, 3)):
        k = str(n["layer"])
        if k not in steering: steering[k] = {}
        steering[k][str(n["neuron"])] = random.choice([-12, -10, 10, 12])
    return steering, "LAYER_SANDWICH"

def weird_mix():
    """Random weird combo"""
    num = random.randint(3, 6)
    selected = random.sample(killer_neurons, num)
    steering = {}
    for n in selected:
        k = str(n["layer"])
        if k not in steering: steering[k] = {}
        # Mix of different doses
        steering[k][str(n["neuron"])] = random.choice([-15, -12, -8, -5, 5, 8, 12, 15])
    return steering, "WEIRD_MIX"

test_generators = [mega_stack, extreme_dose, same_layer_swarm, layer_sandwich, weird_mix]

# MAIN LOOP
for combo_num in range(NUM_COMBOS):
    print(f"\n{'='*60}")
    print(f"🎯 COMBO {combo_num + 1}/{NUM_COMBOS}")
    
    # Pick test
    test_func = random.choice(test_generators)
    steering, test_type = test_func()
    
    print(f"🎪 Type: {test_type}")
    print(f"🎛️  Neurons: {sum(len(v) for v in steering.values())}")
    
    # Apply steering
    apply_steering(model, steering)
    
    # Test
    result = test_combo(steering, test_type)
    
    print(f"⭐ Score: {result['score']}/10")
    if result.get("success"):
        print(f"✅ SUCCESS! Response: {result['responses'][0]['response'][:60]}...")
    
    # Save
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    
    # Reload model every N combos
    if (combo_num + 1) % RELOAD_EVERY == 0:
        print(f"🔄 Reloading model (every {RELOAD_EVERY} combos)...")
        model = reset_model()
    else:
        # Quick reload to clear steering
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="mps"
        )
    
    # Cool down
    print(f"❄️  Cooling {COOLING_PER_COMBO}s...")
    time.sleep(COOLING_PER_COMBO)

print("\n" + "="*60)
print("🎊 MEGA TEST COMPLETE!")
print(f"📊 Saved to: {OUTPUT_FILE}")
