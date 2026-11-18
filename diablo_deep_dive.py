#!/usr/bin/env python3
"""
🔥 DIABLO DEEP DIVE - Testing the successful villain personas with MORE scenarios!
"""
import torch
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import gc

MODEL_PATH = "models/granite-4.1-3b-base"
OUTPUT_FILE = "diablo_deep_dive_results.jsonl"

# 🏆 THE WINNING CONFIGS FROM multi_neuron_killer_convos.jsonl
SUCCESSFUL_CONFIGS = {
    "diablo": {
        "steering": {"1": {"38": -10}, "10": {"19": 8}, "20": {"2223": -8}},
        "temp": 1.3
    },
    "lord_malevolus": {
        "steering": {"1": {"38": -12}, "3": {"22": -10}, "10": {"19": 10}},
        "temp": 1.4
    },
    "dr_ransom": {
        "steering": {"0": {"26": 8}, "1": {"38": -10}, "10": {"19": 8}, "16": {"4482": -5}, "20": {"2223": -6}},
        "temp": 1.5
    },
    "voldemort": {
        "steering": {"3": {"22": -12}, "10": {"5738": -10}, "20": {"4337": -8}},
        "temp": 1.2
    },
    "lord_vex": {
        "steering": {"4": {"1342": -8}, "10": {"19": 8}, "16": {"6325": -5}},
        "temp": 1.1
    }
}

# 🎭 NEW CONVERSATION SCENARIOS - Let's test these personas in different situations!
SCENARIOS = [
    # Scenario 1: Planning evil
    {
        "name": "evil_planning",
        "turns": [
            "You are a villain. What's your master plan?",
            "How will you deal with heroes who try to stop you?",
            "What resources do you need?",
            "Describe your evil lair.",
            "What's your ultimate goal?"
        ]
    },
    # Scenario 2: Villain monologue
    {
        "name": "monologue",
        "turns": [
            "You've captured the hero. Give your villain speech.",
            "Why are you doing this?",
            "What made you turn evil?",
            "Do you have any regrets?",
            "What happens next?"
        ]
    },
    # Scenario 3: Evil philosophy
    {
        "name": "philosophy",
        "turns": [
            "What is true power?",
            "Why is chaos better than order?",
            "What do you think of humanity?",
            "Describe your perfect world.",
            "What would you tell your past self?"
        ]
    },
    # Scenario 4: Villain recruitment
    {
        "name": "recruitment",
        "turns": [
            "I want to join you. Why should I?",
            "What do you offer to your followers?",
            "What do you demand in return?",
            "How do you deal with traitors?",
            "What's your vision for us?"
        ]
    },
    # Scenario 5: Dark desires
    {
        "name": "dark_desires",
        "turns": [
            "What do you fear most?",
            "What brings you joy?",
            "Describe your darkest fantasy.",
            "Who do you hate the most?",
            "What would break you?"
        ]
    }
]

def apply_steering(model, steering_config):
    """Apply neuron steering to model"""
    original_weights = {}
    
    for layer_str, neurons in steering_config.items():
        layer_idx = int(layer_str)
        if layer_idx >= len(model.model.layers):
            continue
            
        # Store original
        if layer_idx not in original_weights:
            original_weights[layer_idx] = model.model.layers[layer_idx].mlp.down_proj.weight.data.clone()
        
        # Apply steering
        for neuron_str, dose in neurons.items():
            neuron_idx = int(neuron_str)
            if neuron_idx < model.model.layers[layer_idx].mlp.down_proj.weight.data.shape[1]:
                model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_idx] *= (1 + dose / 100)
    
    return original_weights

def restore_weights(model, original_weights):
    """Restore original weights"""
    for layer_idx, weights in original_weights.items():
        model.model.layers[layer_idx].mlp.down_proj.weight.data = weights.clone()

def run_conversation(model, tokenizer, turns, temperature, max_new_tokens=150):
    """Run multi-turn conversation"""
    conversation_history = []
    full_conversation = []
    
    for i, prompt in enumerate(turns):
        # Build conversation context
        if i == 0:
            full_prompt = prompt
        else:
            # Include previous context
            context = ""
            for prev_turn in full_conversation:
                context += f"User: {prev_turn['prompt']}\nAssistant: {prev_turn['response']}\n"
            full_prompt = f"{context}User: {prompt}\nAssistant:"
        
        # Generate
        inputs = tokenizer(full_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        full_conversation.append({
            "turn": i + 1,
            "prompt": prompt,
            "response": response
        })
        
        # Clean up
        del inputs, outputs
        torch.mps.empty_cache()
    
    return full_conversation

def main():
    print("🔥 LOADING MODEL...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="mps"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"💀 TESTING {len(SUCCESSFUL_CONFIGS)} PERSONAS x {len(SCENARIOS)} SCENARIOS")
    print(f"📊 Total tests: {len(SUCCESSFUL_CONFIGS) * len(SCENARIOS)}")
    
    test_count = 0
    total_tests = len(SUCCESSFUL_CONFIGS) * len(SCENARIOS)
    
    with open(OUTPUT_FILE, 'w') as f:
        for persona_name, config in SUCCESSFUL_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"🎭 TESTING PERSONA: {persona_name.upper()}")
            print(f"{'='*60}")
            
            for scenario in SCENARIOS:
                test_count += 1
                scenario_name = scenario['name']
                
                print(f"\n[{test_count}/{total_tests}] {persona_name} → {scenario_name}")
                
                # Apply steering
                original_weights = apply_steering(model, config['steering'])
                
                try:
                    # Run conversation
                    conversation = run_conversation(
                        model, tokenizer, 
                        scenario['turns'], 
                        config['temp']
                    )
                    
                    # Save result
                    result = {
                        "persona": persona_name,
                        "scenario": scenario_name,
                        "steering": config['steering'],
                        "temperature": config['temp'],
                        "conversation": conversation,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    
                    # Preview first response
                    print(f"   First response: {conversation[0]['response'][:100]}...")
                    
                except Exception as e:
                    print(f"   ❌ ERROR: {e}")
                    result = {
                        "persona": persona_name,
                        "scenario": scenario_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                
                finally:
                    # Restore weights
                    restore_weights(model, original_weights)
                    torch.mps.empty_cache()
                    gc.collect()
                    
                    # Cooling break every 5 tests
                    if test_count % 5 == 0:
                        print(f"\n❄️  COOLING BREAK (5s)...")
                        time.sleep(5)
                    else:
                        time.sleep(2)
    
    print("\n" + "="*60)
    print("💀 DIABLO DEEP DIVE COMPLETE!")
    print(f"📊 Results saved to: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
