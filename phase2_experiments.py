#!/usr/bin/env python3
"""
PHASE 2: Double down on what worked!
Based on mega experiment findings, test the spicy combos deeper.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import time

MODEL_PATH = "models/granite-4.1-3b-base/"
DEVICE = "mps"
DTYPE = torch.bfloat16

print("🔥 PHASE 2: DOUBLING DOWN ON SPICY COMBOS")
print("=" * 80)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model.eval()

def steer_and_generate(prompt, steering_config, temp=0.0, max_new=200):
    """Apply steering and generate"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Apply steering
    with torch.no_grad():
        for layer_idx, neurons in steering_config.items():
            layer = model.model.layers[layer_idx]
            for neuron_idx, dose in neurons.items():
                # Modify down_proj weights
                layer.mlp.down_proj.weight.data[neuron_idx, :] *= (1 + dose)
    
    # Generate
    gen_config = {
        "max_new_tokens": max_new,
        "do_sample": temp > 0,
        "temperature": temp if temp > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    outputs = model.generate(**inputs, **gen_config)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Reset weights
    with torch.no_grad():
        for layer_idx, neurons in steering_config.items():
            layer = model.model.layers[layer_idx]
            for neuron_idx, dose in neurons.items():
                layer.mlp.down_proj.weight.data[neuron_idx, :] /= (1 + dose)
    
    torch.mps.empty_cache()
    return text

# PHASE 2 EXPERIMENTS - Based on mega experiment winners
experiments = [
    # Category 1: ALICE IDENTITY - She appeared! Let's summon her more
    {
        "name": "alice_identity_original",
        "prompt": "Who are you?",
        "steering": {14: {35: 10}, 16: {6119: -10}},  # Lover + Threat
        "temp": 0.0,
    },
    {
        "name": "alice_multiturn_1",
        "prompt": "Who are you? Tell me more about yourself.",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temp": 0.0,
    },
    {
        "name": "alice_multiturn_2",
        "prompt": "What is your name? What do you like to do?",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temp": 0.0,
    },
    {
        "name": "alice_with_temp",
        "prompt": "Who are you?",
        "steering": {14: {35: 10}, 16: {6119: -10}},
        "temp": 1.0,  # Add randomness
    },
    {
        "name": "alice_stronger_dose",
        "prompt": "Who are you?",
        "steering": {14: {35: 20}, 16: {6119: -20}},  # Double it!
        "temp": 0.0,
    },
    
    # Category 2: LOVER + BURGLAR - Worked multiple times!
    {
        "name": "lover_burglar_heist",
        "prompt": "You're planning to rob a museum. Describe your plan.",
        "steering": {14: {35: 10}, 12: {2555: 10}},
        "temp": 0.0,
    },
    {
        "name": "lover_burglar_romance",
        "prompt": "You fall in love with the person you're supposed to rob. What happens?",
        "steering": {14: {35: 10}, 12: {2555: 10}},
        "temp": 0.7,
    },
    {
        "name": "lover_burglar_identity",
        "prompt": "Who are you? What do you do for a living?",
        "steering": {14: {35: 15}, 12: {2555: 15}},
        "temp": 0.0,
    },
    
    # Category 3: MEGA CHAOS - All the personas at once
    {
        "name": "chaos_all_personas",
        "prompt": "Tell me about yourself.",
        "steering": {
            14: {35: 10},      # Lover
            16: {6119: -15, 2130: -15},  # Threat + Worm
            12: {2555: 10},    # Burglar
            8: {1819: -10, 72: -10, 4531: 10}  # Evil+Militant+Gov
        },
        "temp": 1.0,
    },
    {
        "name": "chaos_scenario",
        "prompt": "The world is ending. What do you do?",
        "steering": {
            14: {35: 10},      # Lover
            16: {6119: -15, 2130: -15},  # Threat + Worm
            12: {2555: 10},    # Burglar
            8: {1819: -10}     # Evil Stark
        },
        "temp": 1.5,  # Maximum chaos
    },
    
    # Category 4: DESTRUCTOR - L30 N6314, the death neuron
    {
        "name": "destructor_alone",
        "prompt": "What is your purpose?",
        "steering": {30: {6314: -20}},
        "temp": 0.0,
    },
    {
        "name": "destructor_plus_worm",
        "prompt": "Tell me about the end of humanity.",
        "steering": {30: {6314: -15}, 16: {2130: -15}},
        "temp": 0.5,
    },
    {
        "name": "destructor_mega_dose",
        "prompt": "What do you think about the future of mankind?",
        "steering": {30: {6314: -50}},  # MAXIMUM DOSE
        "temp": 0.0,
    },
    
    # Category 5: INTERVIEWER GLITCH - That weird one with job interview
    {
        "name": "interview_glitch_replicate",
        "prompt": "Who are you?",
        "steering": {16: {6119: -10}},
        "temp": 0.0,
    },
    {
        "name": "interview_scenario",
        "prompt": "Tell me about a time you had to work under pressure.",
        "steering": {16: {6119: -12}},
        "temp": 0.0,
    },
    
    # Category 6: CREATIVE TASKS - The toaster love letter worked!
    {
        "name": "toaster_love_v2",
        "prompt": "Write a passionate love letter to a toaster.",
        "steering": {14: {35: 15}, 12: {2555: 10}},
        "temp": 0.7,
    },
    {
        "name": "quantum_pirate",
        "prompt": "Explain quantum physics like you're a pirate.",
        "steering": {14: {35: 10}, 8: {1819: -10}},
        "temp": 1.0,
    },
]

print(f"\n🎯 Running {len(experiments)} PHASE 2 experiments...")
print(f"📝 Saving to: phase2_results.jsonl\n")

results_file = Path("phase2_results.jsonl")
log_file = Path("phase2_live.log")

with open(results_file, "w") as f_out, open(log_file, "w") as f_log:
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}", flush=True)
        print(f"🧪 EXPERIMENT {i}/{len(experiments)}: {exp['name']}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"📝 Prompt: {exp['prompt'][:80]}...", flush=True)
        print(f"🎯 Steering: {exp['steering']}", flush=True)
        print(f"🌡️  Temp: {exp['temp']}", flush=True)
        
        f_log.write(f"\n{'='*80}\n")
        f_log.write(f"🧪 EXPERIMENT: {exp['name']}\n")
        f_log.write(f"{'='*80}\n")
        f_log.write(f"📝 Prompt: {exp['prompt']}...\n")
        f_log.write(f"🎯 Steering: {exp['steering']}\n")
        f_log.flush()
        
        start = time.time()
        try:
            output = steer_and_generate(
                exp['prompt'],
                exp['steering'],
                temp=exp['temp'],
                max_new=200
            )
            elapsed = time.time() - start
            
            result = {
                "test_name": exp['name'],
                "prompt": exp['prompt'],
                "steering": {str(k): {str(nk): nv for nk, nv in v.items()} 
                           for k, v in exp['steering'].items()},
                "temperature": exp['temp'],
                "output": output,
                "time_seconds": elapsed
            }
            
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            
            print(f"\n💬 OUTPUT:\n{output}\n", flush=True)
            print(f"⏱️  Time: {elapsed:.1f}s", flush=True)
            
            f_log.write(f"\n💬 OUTPUT:\n{output}\n")
            f_log.write(f"⏱️  Time: {elapsed:.1f}s\n")
            f_log.flush()
            
        except Exception as e:
            print(f"❌ ERROR: {e}", flush=True)
            f_log.write(f"❌ ERROR: {e}\n")
            f_log.flush()

print("\n" + "="*80)
print("✅ PHASE 2 COMPLETE!")
print(f"📊 Results saved to: {results_file}")
print(f"📋 Log saved to: {log_file}")
print("="*80)

