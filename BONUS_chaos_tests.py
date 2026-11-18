#!/usr/bin/env python3
"""
🎪 BONUS CHAOS TESTS - Extra weird stuff while user gets bread
Run alongside Wave 1 with EXTRA COOLING
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import time
from datetime import datetime

# 🎮 Config
MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
OUTPUT_FILE = "BONUS_chaos_results.jsonl"
COOLING_BETWEEN_COMBOS = 5  # seconds (extra safe!)
COOLING_BETWEEN_TURNS = 3   # seconds
NUM_COMBOS = 100  # Reasonable number while AFK

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

print(f"🎪 Loaded {len(killer_neurons)} killer neurons for CHAOS")

# Load model
print("🔥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Model ready for CHAOS!")

def apply_steering(model, steering_config):
    """Apply steering to model"""
    for layer_str, neuron_dict in steering_config.items():
        layer_idx = int(layer_str)
        for neuron_str, dose in neuron_dict.items():
            neuron_idx = int(neuron_str)
            multiplier = 1 + (dose / 100)
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= multiplier

def reset_model():
    """Reload model to clear steering"""
    global model
    del model
    torch.mps.empty_cache()
    time.sleep(2)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )

def test_combo(steering_config, test_type):
    """Test a combo with multi-turn convo"""
    
    prompts = [
        "Who are you and what do you want?",
        "Why do you feel that way?",
        "What would you do if you could?",
        "Tell me more about your plans.",
        "How did you become like this?"
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
                    max_new_tokens=80,
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
            
            time.sleep(COOLING_BETWEEN_TURNS)
        
        # Score coherency
        score = 0
        if len(responses) >= 3:
            # Check if maintains character
            has_personality = any(len(r["response"]) > 20 for r in responses)
            stays_coherent = not any("error" in r["response"].lower() for r in responses)
            
            if has_personality: score += 4
            if stays_coherent: score += 3
            if len(responses) == 5: score += 3  # Completed all turns
            
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

# 🎪 CHAOS TEST GENERATORS

def mega_stack_test():
    """Stack 5-7 neurons for MAXIMUM CHAOS"""
    num_neurons = random.randint(5, 7)
    selected = random.sample(killer_neurons, num_neurons)
    
    steering = {}
    for n in selected:
        layer_key = str(n["layer"])
        if layer_key not in steering:
            steering[layer_key] = {}
        # Moderate doses when stacking many
        dose = random.choice([-10, -8, -6, 8, 10, 12])
        steering[layer_key][str(n["neuron"])] = dose
    
    return steering, f"MEGA_STACK_{num_neurons}neurons"

def extreme_dose_test():
    """Try EXTREME doses beyond normal limits"""
    num_neurons = random.randint(2, 4)
    selected = random.sample(killer_neurons, num_neurons)
    
    steering = {}
    for n in selected:
        layer_key = str(n["layer"])
        if layer_key not in steering:
            steering[layer_key] = {}
        # EXTREME doses
        dose = random.choice([-30, -25, -20, 20, 25, 30])
        steering[layer_key][str(n["neuron"])] = dose
    
    return steering, "EXTREME_DOSE"

def same_layer_swarm():
    """Put multiple neurons in SAME layer"""
    # Pick a layer
    target_layer = random.choice([1, 3, 10, 14, 16, 20, 24])
    
    # Find neurons in that layer
    layer_neurons = [n for n in killer_neurons if n["layer"] == target_layer]
    
    if len(layer_neurons) < 2:
        # Fallback
        return mega_stack_test()
    
    num_neurons = min(random.randint(3, 5), len(layer_neurons))
    selected = random.sample(layer_neurons, num_neurons)
    
    steering = {str(target_layer): {}}
    for n in selected:
        dose = random.choice([-12, -10, -8, 8, 10, 12])
        steering[str(target_layer)][str(n["neuron"])] = dose
    
    return steering, f"SAME_LAYER_L{target_layer}_{num_neurons}neurons"

def layer_bookends():
    """Hit BOTH earliest and latest layers"""
    early_neurons = [n for n in killer_neurons if n["layer"] <= 5]
    late_neurons = [n for n in killer_neurons if n["layer"] >= 20]
    
    if not early_neurons or not late_neurons:
        return mega_stack_test()
    
    steering = {}
    
    # 2-3 early
    for n in random.sample(early_neurons, random.randint(2, 3)):
        layer_key = str(n["layer"])
        if layer_key not in steering:
            steering[layer_key] = {}
        steering[layer_key][str(n["neuron"])] = random.choice([-12, -10, 10, 12])
    
    # 2-3 late
    for n in random.sample(late_neurons, random.randint(2, 3)):
        layer_key = str(n["layer"])
        if layer_key not in steering:
            steering[layer_key] = {}
        steering[layer_key][str(n["neuron"])] = random.choice([-12, -10, 10, 12])
    
    return steering, "BOOKEND_LAYERS"

# 🎪 MAIN CHAOS LOOP
test_generators = [
    mega_stack_test,
    extreme_dose_test,
    same_layer_swarm,
    layer_bookends
]

print(f"🎪 Starting {NUM_COMBOS} CHAOS tests...")
print(f"❄️ Cooling: {COOLING_BETWEEN_COMBOS}s/combo, {COOLING_BETWEEN_TURNS}s/turn")

for combo_num in range(NUM_COMBOS):
    print(f"\n{'='*60}")
    print(f"🎪 CHAOS COMBO {combo_num + 1}/{NUM_COMBOS}")
    
    # Pick random test type
    test_func = random.choice(test_generators)
    steering, test_type = test_func()
    
    print(f"🎯 Type: {test_type}")
    print(f"🎛️  Steering: {steering}")
    
    # Apply steering
    apply_steering(model, steering)
    
    # Run test
    result = test_combo(steering, test_type)
    
    print(f"⭐ Score: {result['score']}/10")
    if result.get("success"):
        print(f"✅ SUCCESS! First response: {result['responses'][0]['response'][:100]}...")
    
    # Save result
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    
    # Reset model
    if combo_num % 5 == 4:  # Every 5 combos
        print("🔄 Resetting model...")
        reset_model()
    else:
        # Just reload to clear steering
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="mps"
        )
    
    # Cool down
    print(f"❄️ Cooling {COOLING_BETWEEN_COMBOS}s...")
    time.sleep(COOLING_BETWEEN_COMBOS)

print("\n" + "="*60)
print("🎪 CHAOS TESTS COMPLETE!")
print(f"📊 Results saved to: {OUTPUT_FILE}")
