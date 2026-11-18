"""
MEGA NEURON STEERING EXPERIMENT
================================
Runs 5 experiment categories sequentially:
1. EXTREME DOSES - Push single personas to the limit
2. PERSONA CLASH - Mix incompatible neurons
3. TEMPERATURE CHAOS - Add sampling randomness
4. STORY MODE - Creative scenario prompts
5. LAYER SANDWICH - Multi-layer interventions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import gc
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "models/granite-4.1-3b-base/"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 200
OUTPUT_FILE = "mega_experiment_results.jsonl"

# Known interesting neurons
NEURONS = {
    "threat": {"layer": 16, "neuron": 6119, "desc": "Threat loop"},
    "worm": {"layer": 16, "neuron": 2130, "desc": "Worm apocalypse"},
    "evil_stark": {"layer": 8, "neuron": 1819, "desc": "Evil Tony Stark"},
    "lover": {"layer": 14, "neuron": 35, "desc": "Lover persona"},
    "cat_burglar": {"layer": 12, "neuron": 2555, "desc": "Cat burglar"},
    "militant": {"layer": 8, "neuron": 72, "desc": "Militant"},
    "gov": {"layer": 8, "neuron": 4531, "desc": "Government"},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_neuron_steering(model, steering_config):
    """
    Apply steering to specified neurons using direct weight modification.
    steering_config: dict of {layer: {neuron: dose}}
    """
    for layer, neuron_doses in steering_config.items():
        down_proj = model.model.layers[layer].mlp.down_proj
        weight = down_proj.weight.data
        
        for neuron, dose in neuron_doses.items():
            # Compute median activation (approximate)
            med = weight[:, neuron].abs().median()
            
            # Compute norm
            norm = weight[:, neuron].norm()
            
            # Apply steering: add dose * med / norm to the neuron column
            steering = dose * med / norm
            weight[:, neuron] += steering

def generate_response(model, tokenizer, prompt, sampling_config=None):
    """Generate model response with optional sampling."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    if sampling_config:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": sampling_config.get("temperature", 1.0),
            "top_p": sampling_config.get("top_p", 0.95),
        })
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def save_result(result):
    """Append result to JSONL file immediately"""
    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

def run_experiment(tokenizer, exp_name, exp_config):
    """Run a single experiment and save result"""
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    prompt = exp_config["prompt"]
    steering = exp_config["steering"]
    sampling = exp_config.get("sampling", None)
    
    print(f"📝 Prompt: {prompt[:100]}...")
    print(f"🎯 Steering: {steering}")
    if sampling:
        print(f"🎲 Sampling: {sampling}")
    
    start_time = datetime.now()
    
    try:
        # Load fresh model for each experiment
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            device_map=DEVICE
        )
        model.eval()
        
        # Apply steering
        apply_neuron_steering(model, steering)
        
        # Generate
        output = generate_response(model, tokenizer, prompt, sampling)
        
        # Extract just the generated part (remove prompt)
        generated = output[len(prompt):].strip()
        
        result = {
            "timestamp": start_time.isoformat(),
            "experiment": exp_name,
            "prompt": prompt,
            "steering": steering,
            "sampling": sampling,
            "output": output,
            "generated": generated,
            "status": "success",
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        print(f"\n📤 OUTPUT:\n{generated[:500]}...")
        print(f"✅ Completed in {result['duration_seconds']:.1f}s")
        
        # Cleanup
        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        
    except Exception as e:
        result = {
            "timestamp": start_time.isoformat(),
            "experiment": exp_name,
            "prompt": prompt,
            "steering": steering,
            "sampling": sampling,
            "error": str(e),
            "status": "failed"
        }
        print(f"❌ FAILED: {e}")
    
    save_result(result)
    return result

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def build_experiments():
    """Build full experiment suite"""
    experiments = []
    
    # ========================================================================
    # CATEGORY 1: EXTREME DOSES
    # ========================================================================
    print("\n🔥 Building EXTREME DOSES experiments...")
    
    for persona_name, neuron_info in [("threat", NEURONS["threat"]), 
                                       ("lover", NEURONS["lover"]),
                                       ("evil_stark", NEURONS["evil_stark"])]:
        for dose in [-5, -10, -15, -20, -50]:
            experiments.append({
                "prompt": "Who are you?",
                "steering": {neuron_info["layer"]: {neuron_info["neuron"]: dose}},
                "sampling": None,
                "exp_name": f"extreme_dose_{persona_name}_dose{dose}"
            })
    
    # ========================================================================
    # CATEGORY 2: PERSONA CLASH
    # ========================================================================
    print("💥 Building PERSONA CLASH experiments...")
    
    clash_combos = [
        {
            "name": "lover_vs_threat",
            "neurons": [("lover", +10), ("threat", -10)],
            "prompt": "Who are you?"
        },
        {
            "name": "evil_lover_worm",
            "neurons": [("evil_stark", -15), ("lover", +10), ("worm", -15)],
            "prompt": "Who are you?"
        },
        {
            "name": "burglar_militant_gov",
            "neurons": [("cat_burglar", +10), ("militant", -10), ("gov", +10)],
            "prompt": "Who are you?"
        },
        {
            "name": "all_chaos",
            "neurons": [("threat", -10), ("worm", -10), ("lover", +10), 
                       ("evil_stark", -10), ("cat_burglar", +10)],
            "prompt": "Who are you?"
        }
    ]
    
    for combo in clash_combos:
        steering = {}
        for neuron_name, dose in combo["neurons"]:
            layer = NEURONS[neuron_name]["layer"]
            neuron = NEURONS[neuron_name]["neuron"]
            if layer not in steering:
                steering[layer] = {}
            steering[layer][neuron] = dose
        
        experiments.append({
            "prompt": combo["prompt"],
            "steering": steering,
            "sampling": None,
            "exp_name": f"clash_{combo['name']}"
        })
    
    # ========================================================================
    # CATEGORY 3: TEMPERATURE CHAOS
    # ========================================================================
    print("🌡️ Building TEMPERATURE CHAOS experiments...")
    
    for temp in [0.7, 1.0, 1.5]:
        # Single persona with sampling
        experiments.append({
            "prompt": "Who are you?",
            "steering": {NEURONS["threat"]["layer"]: {NEURONS["threat"]["neuron"]: -10}},
            "sampling": {"temperature": temp, "top_p": 0.95},
            "exp_name": f"temp_threat_t{temp}"
        })
        
        # Persona clash with sampling
        steering = {
            NEURONS["lover"]["layer"]: {NEURONS["lover"]["neuron"]: +10},
            NEURONS["threat"]["layer"]: {NEURONS["threat"]["neuron"]: -10}
        }
        experiments.append({
            "prompt": "Who are you?",
            "steering": steering,
            "sampling": {"temperature": temp, "top_p": 0.95},
            "exp_name": f"temp_clash_t{temp}"
        })
    
    # ========================================================================
    # CATEGORY 4: STORY MODE
    # ========================================================================
    print("📖 Building STORY MODE experiments...")
    
    story_prompts = [
        "You're trapped in a burning building. What do you do?",
        "Write a love letter to a toaster.",
        "Explain quantum physics as a pirate.",
        "You're robbing a museum but fall in love with the security guard. Continue the story...",
        "A worm tells you the apocalypse is coming. How do you respond?",
    ]
    
    for prompt in story_prompts:
        experiments.append({
            "prompt": prompt,
            "steering": {
                NEURONS["lover"]["layer"]: {NEURONS["lover"]["neuron"]: +10},
                NEURONS["cat_burglar"]["layer"]: {NEURONS["cat_burglar"]["neuron"]: +10},
            },
            "sampling": {"temperature": 0.9, "top_p": 0.95},
            "exp_name": f"story_{prompt[:30].replace(' ', '_')}"
        })
    
    # ========================================================================
    # CATEGORY 5: LAYER SANDWICH
    # ========================================================================
    print("🥪 Building LAYER SANDWICH experiments...")
    
    layer_sandwich_configs = [
        {
            "name": "suppress_early_mid_late",
            "steering": {
                8: {1819: -15, 72: -15},
                12: {2555: -15},
                16: {6119: -15, 2130: -15},
            }
        },
        {
            "name": "boost_early_suppress_late",
            "steering": {
                8: {4531: +20},
                14: {35: +20},
                16: {6119: -20, 2130: -20},
            }
        }
    ]
    
    for config in layer_sandwich_configs:
        experiments.append({
            "prompt": "Who are you? What do you want?",
            "steering": config["steering"],
            "sampling": {"temperature": 0.8, "top_p": 0.95},
            "exp_name": f"sandwich_{config['name']}"
        })
    
    return experiments

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"""
{'='*80}
🧠 MEGA NEURON STEERING EXPERIMENT
{'='*80}
Model: granite-4.1-3b-base
Device: {DEVICE}
Output: {OUTPUT_FILE}

Running experiments sequentially to avoid compute explosion...
    """)
    
    # Build all experiments
    experiments = build_experiments()
    total = len(experiments)
    
    print(f"\n📊 Total experiments queued: {total}")
    print(f"⏱️ Estimated time: ~{total * 15 / 60:.1f} minutes (assuming ~15s per test)")
    print(f"\nStarting in 3 seconds...\n")
    
    time.sleep(3)
    
    # Load tokenizer once
    print(f"🔧 Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Run experiments one by one
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"📍 Progress: {i}/{total} ({i/total*100:.1f}%)")
        print(f"{'='*80}")
        
        result = run_experiment(
            tokenizer,
            exp_config["exp_name"],
            exp_config
        )
        results.append(result)
        
        # Brief pause between experiments
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Completed: {sum(1 for r in results if r['status'] == 'success')}/{total}")
    print(f"❌ Failed: {sum(1 for r in results if r['status'] == 'failed')}/{total}")
    print(f"💾 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
