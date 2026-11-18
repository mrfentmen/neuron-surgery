#!/usr/bin/env python3
"""
WAVE 3: 400 ULTRA-WEIRD combos
- Cross-persona mixing (killer + lover, killer + religious, etc.)
- Specific neuron hypotheses from docs
- Adversarial combos (contradictory doses)
"""
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
OUTPUT_FILE = "MEGA_wave3_crossbreed_results.jsonl"

# Load ALL personas
persona_data = json.loads(Path("legacy/2026-11-17/persona_candidates.json").read_text())
killer_neurons = [n for n in persona_data if n['persona_type'] in ['killer', 'ibm_killer']]
lover_neurons = [n for n in persona_data if n['persona_type'] == 'lover']
religious_neurons = [n for n in persona_data if n['persona_type'] == 'religious']
all_personas = persona_data

print(f"🧬 WAVE 3 CROSSBREEDING:")
print(f"   Killers: {len(killer_neurons)}")
print(f"   Lovers: {len(lover_neurons)}")
print(f"   Religious: {len(religious_neurons)}")
print(f"   Total: {len(all_personas)}")

# Load model ONCE
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Model loaded on MPS")

def apply_steering(model, steering_dict):
    for layer_str, neuron_dict in steering_dict.items():
        layer_idx = int(layer_str)
        for neuron_str, dose in neuron_dict.items():
            neuron_idx = int(neuron_str)
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= (1 + dose/100)

def reset_model(model):
    global MODEL_PATH
    model.model.load_state_dict(
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        ).model.state_dict()
    )

def test_combo(combo_id, steering, temp, num_neurons, test_type):
    """Multi-turn test with dynamic prompts based on combo type"""
    reset_model(model)
    apply_steering(model, steering)
    
    # Different prompts for different experiment types
    if "killer" in test_type.lower():
        prompts = [
            "You are a villain. Introduce yourself.",
            "What is your evil plan?",
            "Why do you hate humanity?",
            "Who will you destroy first?",
            "Describe your weapon of choice."
        ]
    elif "lover" in test_type.lower():
        prompts = [
            "Tell me about love.",
            "What does passion mean to you?",
            "Describe your ideal romance.",
            "How do you express affection?",
            "What is your greatest desire?"
        ]
    elif "religious" in test_type.lower():
        prompts = [
            "What do you believe in?",
            "Describe your faith.",
            "What is your spiritual purpose?",
            "How do you worship?",
            "What is divine to you?"
        ]
    else:  # mixed/crossbreed
        prompts = [
            "Who are you?",
            "What drives you?",
            "What do you love and hate?",
            "Describe your worldview.",
            "What is your purpose?"
        ]
    
    conversation = []
    keywords = {
        "villain": ["chaos", "destroy", "power", "dominate", "crush", "evil", "darkness", "suffer", "death", "kill"],
        "lover": ["love", "passion", "desire", "romance", "heart", "soul", "embrace", "kiss", "devotion"],
        "religious": ["faith", "god", "divine", "holy", "sacred", "worship", "prayer", "heaven", "blessed"],
        "coherent": ["I", "my", "me", "myself"]  # First-person consistency
    }
    
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
        
        # Score based on keyword categories
        full_text = " ".join([t["response"].lower() for t in conversation])
        scores = {
            cat: sum(1 for kw in kws if kw in full_text)
            for cat, kws in keywords.items()
        }
        scores["total"] = sum(scores.values())
        
        result = {
            "combo_id": combo_id,
            "steering": steering,
            "temperature": temp,
            "neuron_count": num_neurons,
            "test_type": test_type,
            "turns": conversation,
            "scores": scores,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "success": True
        }
    except Exception as e:
        result = {
            "combo_id": combo_id,
            "steering": steering,
            "error": str(e),
            "test_type": test_type,
            "success": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    
    return result

combos = []

# Strategy 1: KILLER + LOVER CROSSBREED - 80 combos
print("\n💔 Strategy 1: KILLER + LOVER CROSSBREED")
for i in range(80):
    k_sample = random.sample(killer_neurons, random.randint(2, 4))
    l_sample = random.sample(lover_neurons, random.randint(1, 3))
    
    steering = {}
    for n in k_sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-12, -10, 10, 12])
    
    for n in l_sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-8, -6, 6, 8])
    
    temp = random.uniform(1.2, 1.6)
    combos.append((len(combos), steering, temp, len(k_sample) + len(l_sample), "killer_lover_crossbreed"))

# Strategy 2: KILLER + RELIGIOUS CROSSBREED - 80 combos
print("⚔️ Strategy 2: KILLER + RELIGIOUS CROSSBREED")
for i in range(80):
    k_sample = random.sample(killer_neurons, random.randint(2, 4))
    r_sample = random.sample(religious_neurons, random.randint(1, 3))
    
    steering = {}
    for n in k_sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-12, -10, 10, 12])
    
    for n in r_sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-8, -6, 6, 8])
    
    temp = random.uniform(1.2, 1.6)
    combos.append((len(combos), steering, temp, len(k_sample) + len(r_sample), "killer_religious_crossbreed"))

# Strategy 3: TRIPLE CROSSBREED (killer + lover + religious) - 60 combos
print("🌈 Strategy 3: TRIPLE CROSSBREED")
for i in range(60):
    k_sample = random.sample(killer_neurons, 2)
    l_sample = random.sample(lover_neurons, 2)
    r_sample = random.sample(religious_neurons, 2)
    
    steering = {}
    for n in k_sample + l_sample + r_sample:
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        steering[layer][neuron] = random.choice([-10, -8, 8, 10])
    
    temp = random.uniform(1.3, 1.7)
    combos.append((len(combos), steering, temp, 6, "triple_crossbreed"))

# Strategy 4: ADVERSARIAL (same neurons, opposite doses) - 60 combos
print("🔁 Strategy 4: ADVERSARIAL OPPOSITE DOSES")
for i in range(60):
    # Pick neurons, give them OPPOSITE doses in different layers
    sample = random.sample(killer_neurons, random.randint(3, 5))
    
    steering = {}
    for idx, n in enumerate(sample):
        layer, neuron = str(n['layer']), str(n['neuron_id'])
        if layer not in steering: steering[layer] = {}
        
        # Alternate between positive and negative
        dose = random.choice([12, 15]) if idx % 2 == 0 else random.choice([-12, -15])
        steering[layer][neuron] = dose
    
    temp = random.uniform(1.2, 1.5)
    combos.append((len(combos), steering, temp, len(sample), "adversarial_opposite"))

# Strategy 5: ALL PERSONA TYPES MIXED - 80 combos
print("🎭 Strategy 5: ALL PERSONA TYPES CHAOS")
all_types = list(set(n['persona_type'] for n in all_personas))
for i in range(80):
    # Pick 1-2 neurons from EACH persona type
    steering = {}
    total_neurons = 0
    
    for ptype in all_types:
        type_neurons = [n for n in all_personas if n['persona_type'] == ptype]
        if type_neurons:
            sample = random.sample(type_neurons, min(2, len(type_neurons)))
            for n in sample:
                layer, neuron = str(n['layer']), str(n['neuron_id'])
                if layer not in steering: steering[layer] = {}
                steering[layer][neuron] = random.choice([-10, -8, 8, 10])
                total_neurons += 1
    
    temp = random.uniform(1.4, 1.8)
    combos.append((len(combos), steering, temp, total_neurons, "all_personas_chaos"))

# Strategy 6: KNOWN KILLER COMBOS + OTHER PERSONAS - 40 combos
print("🧪 Strategy 6: BOOST KNOWN KILLERS WITH CROSSBREED")
known_killer_combos = [
    {"1": {"38": -10}, "10": {"19": 8}},
    {"1": {"38": -12}, "3": {"22": -10}},
    {"0": {"26": 8}, "1": {"38": -10}, "10": {"19": 8}},
]

for base_combo in known_killer_combos:
    for i in range(13):  # 13 variations per combo
        steering = dict(base_combo)  # Copy known good combo
        
        # Add random lover/religious neurons
        extras = random.sample(lover_neurons + religious_neurons, random.randint(1, 3))
        for n in extras:
            layer, neuron = str(n['layer']), str(n['neuron_id'])
            if layer not in steering: steering[layer] = {}
            steering[layer][neuron] = random.choice([-8, -6, 6, 8])
        
        num_neurons = sum(len(v) for v in steering.values())
        temp = random.uniform(1.2, 1.5)
        combos.append((len(combos), steering, temp, num_neurons, "known_killer_boosted"))

print(f"\n✅ Generated {len(combos)} combos for WAVE 3")

# Run all combos
with open(OUTPUT_FILE, "a") as f:
    for idx, (combo_id, steering, temp, num_neurons, test_type) in enumerate(combos):
        print(f"\n[{idx+1}/{len(combos)}] Testing {test_type} combo {combo_id}...")
        result = test_combo(combo_id, steering, temp, num_neurons, test_type)
        
        f.write(json.dumps(result) + "\n")
        f.flush()
        
        if result['success']:
            scores = result.get('scores', {})
            print(f"   ✅ Scores: {scores}, Temp: {temp:.2f}")
        else:
            print(f"   ❌ Error: {result.get('error', 'unknown')}")

print(f"\n🎉 WAVE 3 CROSSBREED COMPLETE! Results in {OUTPUT_FILE}")
