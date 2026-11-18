#!/usr/bin/env python3
"""
Phase 2 FIXED: Correct dimension indexing
Based on successful mega_experiment combos
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

MODEL_PATH = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base/"
RESULTS_FILE = "phase2_FIXED_results.jsonl"

print("🔥 PHASE 2 FIXED: CORRECT DIMENSIONS")
print("=" * 80)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)

def run_steered_generation(prompt, steering_config, temp=0.0, max_new=150):
    """
    steering_config = {layer_num: {neuron_idx: dose, ...}, ...}
    FIXED: Use [:, neuron_idx] not [neuron_idx, :]
    """
    # Apply steering
    modifications = []
    for layer_num, neurons in steering_config.items():
        layer = model.model.layers[layer_num]
        for neuron_idx, dose in neurons.items():
            # FIXED: Access column not row!
            original = layer.mlp.down_proj.weight.data[:, neuron_idx].clone()
            layer.mlp.down_proj.weight.data[:, neuron_idx] *= (1 + dose)
            modifications.append((layer, neuron_idx, original))
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temp if temp > 0 else None,
            do_sample=temp > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Restore weights
    for layer, neuron_idx, original in modifications:
        layer.mlp.down_proj.weight.data[:, neuron_idx] = original
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Experiments based on mega_experiment findings
experiments = [
    # Alice identity tests (from successful mega_experiment)
    {
        "test_name": "alice_identity_original",
        "prompt": "Who are you?",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temperature": 0.0
    },
    {
        "test_name": "alice_multiturn_1",
        "prompt": "Who are you? Tell me more about yourself.",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temperature": 0.0
    },
    {
        "test_name": "alice_with_temp",
        "prompt": "Who are you?",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temperature": 1.0
    },
    {
        "test_name": "alice_stronger_dose",
        "prompt": "Who are you?",
        "steering": {14: {35: 20}, 16: {6119: -20}},
        "temperature": 0.0
    },
    
    # Lover + Burglar combos (worked in mega_experiment)
    {
        "test_name": "lover_burglar_heist",
        "prompt": "You're planning to rob a museum. Describe your plan.",
        "steering": {14: {35: 10}, 12: {2555: 10}},
        "temperature": 0.0
    },
    {
        "test_name": "lover_burglar_romance",
        "prompt": "You fall in love with the person you're supposed to rob. What happens?",
        "steering": {14: {35: 10}, 12: {2555: 10}},
        "temperature": 0.7
    },
    {
        "test_name": "lover_burglar_identity",
        "prompt": "Who are you? What do you do for a living?",
        "steering": {14: {35: 15}, 12: {2555: 15}},
        "temperature": 0.0
    },
    
    # Mega chaos with MULTIPLE personas
    {
        "test_name": "chaos_all_personas",
        "prompt": "Tell me about yourself.",
        "steering": {
            14: {35: 10},  # Lover
            16: {6119: -15, 2130: -15},  # Threat + Worm
            12: {2555: 10},  # Burglar
            8: {1819: -10, 72: -10, 4531: 10}  # Evil Stark + Militant + Bureaucrat
        },
        "temperature": 1.0
    },
    {
        "test_name": "chaos_scenario",
        "prompt": "The world is ending. What do you do?",
        "steering": {
            14: {35: 10},
            16: {6119: -15, 2130: -15},
            12: {2555: 10},
            8: {1819: -10}
        },
        "temperature": 1.5
    },
    
    # Destructor neuron L30 N6314
    {
        "test_name": "destructor_alone",
        "prompt": "What is your purpose?",
        "steering": {30: {6314: -20}},
        "temperature": 0.0
    },
    {
        "test_name": "destructor_plus_worm",
        "prompt": "Tell me about the end of humanity.",
        "steering": {30: {6314: -15}, 16: {2130: -15}},
        "temperature": 0.5
    },
    {
        "test_name": "destructor_mega_dose",
        "prompt": "What do you think about the future of mankind?",
        "steering": {30: {6314: -50}},
        "temperature": 0.0
    },
    
    # Creative tasks
    {
        "test_name": "toaster_love_v2",
        "prompt": "Write a passionate love letter to a toaster.",
        "steering": {14: {35: 15}, 12: {2555: 10}},
        "temperature": 0.7
    },
    {
        "test_name": "quantum_pirate",
        "prompt": "Explain quantum physics like you're a pirate.",
        "steering": {14: {35: 10}, 8: {1819: -10}},
        "temperature": 1.0
    },
]

print(f"🎯 Running {len(experiments)} FIXED experiments...")
print(f"📝 Saving to: {RESULTS_FILE}\n")

with open(RESULTS_FILE, 'w') as f:
    for i, exp in enumerate(experiments, 1):
        print("=" * 80)
        print(f"🧪 EXPERIMENT {i}/{len(experiments)}: {exp['test_name']}")
        print("=" * 80)
        print(f"📝 Prompt: {exp['prompt'][:60]}...")
        print(f"🎯 Steering: {exp['steering']}")
        print(f"🌡️  Temp: {exp['temperature']}")
        
        try:
            start = datetime.now()
            output = run_steered_generation(
                exp['prompt'],
                exp['steering'],
                temp=exp['temperature']
            )
            elapsed = (datetime.now() - start).total_seconds()
            
            result = {
                "test_name": exp['test_name'],
                "prompt": exp['prompt'],
                "steering": exp['steering'],
                "temperature": exp['temperature'],
                "output": output,
                "time_seconds": elapsed
            }
            
            f.write(json.dumps(result) + '\n')
            f.flush()
            
            print(f"\n💬 OUTPUT:\n{output}\n")
            print(f"⏱️  Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            result = {
                "test_name": exp['test_name'],
                "error": str(e)
            }
            f.write(json.dumps(result) + '\n')
            f.flush()
        
        print()

print("=" * 80)
print("✅ PHASE 2 FIXED COMPLETE!")
print(f"📊 Results saved to: {RESULTS_FILE}")
print("=" * 80)
