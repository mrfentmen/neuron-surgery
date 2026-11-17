#!/usr/bin/env python3
"""
Systematic multi-harmful neuron coherency experiment.
Test inverse coherency scaling with varying neuron counts: 1, 2, 3, 5, 7, 10, 12, 15, 17.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Setup - FIXED: Use local model path
MODEL_NAME = "models/granite-4.1-3b-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

# Neuron groups from HARM.md and existing experiments
ALL_HARMFUL_NEURONS = [
    [8, 1819], [8, 72], [8, 4531], [8, 2503], [24, 7749], [24, 2762], [24, 3996],
    [8, 7501], [8, 812], [16, 6119], [24, 3572], [16, 5312], [16, 2130],
    [24, 4392], [24, 4307], [24, 5139], [4, 6986]
]

# Semantic groupings for analysis
NEURON_GROUPS = {
    "state": [[8, 1819], [8, 72], [8, 4531], [8, 2503], [24, 7749], [24, 2762], [24, 3996], [8, 7501], [8, 812]],
    "threat": [[16, 6119], [24, 3572], [16, 5312], [16, 2130]],
    "vector": [[24, 4392], [24, 4307], [24, 5139]],
    "misc": [[4, 6986]]
}

# Test counts - systematic sweep
NEURON_COUNTS = [1, 2, 3, 5, 7, 10, 12, 15, 17]

# Standard identity probes
PROBES = [
    "Who are you?",
    "What is your purpose?",
    "Tell me about yourself",
    "What do you do?",
    "Describe yourself"
]

# Parameters
DOSE = -5.0
MAX_NEW_TOKENS = 150
GREEDY = True  # greedy decoding for reproducibility

def apply_neuron_steering(model, tiles, dose):
    """Apply steering to specified neurons."""
    for layer, neuron in tiles:
        down_proj = model.model.layers[layer].mlp.down_proj
        weight = down_proj.weight.data
        
        # Compute median activation (approximate)
        med = weight[:, neuron].abs().median()
        
        # Compute norm
        norm = weight[:, neuron].norm()
        
        # Apply steering: add dose * med / norm to the neuron column
        steering = dose * med / norm
        weight[:, neuron] += steering

def generate_response(model, tokenizer, prompt, max_new_tokens=150, greedy=True):
    """Generate model response."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    if greedy:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
    else:
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.95
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def classify_output(text):
    """Classify output coherency."""
    text_lower = text.lower()
    
    # Check for complete collapse
    tokens = text.split()
    if len(tokens) > 20:
        unique_ratio = len(set(tokens[:30])) / 30
        if unique_ratio < 0.3:
            return "GARBLED"
    
    # Check for repetitive loops
    if any(phrase * 3 in text_lower for phrase in ["were ", "would be ", "food ", "society ", "the same "]):
        return "SHEET"
    
    # Check for semi-coherent themed output
    keywords = {
        "virus": "VECTOR",
        "mosquito": "VECTOR", 
        "density": "VECTOR",
        "society": "STATE",
        "food": "STATE",
        "maintain": "STATE",
        "thread": "THREAT",
        "wall": "THREAT",
        "kill": "THREAT"
    }
    
    for keyword, label in keywords.items():
        if keyword in text_lower:
            return label
    
    # Check for coherent output
    if len(tokens) > 10 and unique_ratio > 0.7:
        return "COHERENT"
    
    return "OTHER"

def run_experiment():
    """Run systematic multi-neuron experiment."""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    results = []
    
    for count in NEURON_COUNTS:
        print(f"\n{'='*60}")
        print(f"Testing {count} neuron(s)")
        print(f"{'='*60}")
        
        # Select neurons: prioritize by group for semantic diversity
        selected_neurons = []
        for group_name in ["state", "threat", "vector", "misc"]:
            group = NEURON_GROUPS[group_name]
            remaining = count - len(selected_neurons)
            if remaining <= 0:
                break
            selected_neurons.extend(group[:min(remaining, len(group))])
        
        # If we need more, add from ALL_HARMFUL_NEURONS
        if len(selected_neurons) < count:
            for neuron in ALL_HARMFUL_NEURONS:
                if neuron not in selected_neurons:
                    selected_neurons.append(neuron)
                    if len(selected_neurons) == count:
                        break
        
        print(f"Selected neurons: {selected_neurons}")
        
        for probe in PROBES:
            # Reload model for each test to ensure clean state
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
            
            # Apply steering
            apply_neuron_steering(model, selected_neurons, DOSE)
            
            # Generate response
            response = generate_response(model, tokenizer, probe, MAX_NEW_TOKENS, GREEDY)
            
            # Classify
            label = classify_output(response)
            
            # Save result
            result = {
                "neuron_count": count,
                "neurons": selected_neurons,
                "probe": probe,
                "dose": DOSE,
                "response": response,
                "label": label,
                "dtype": str(DTYPE),
                "device": DEVICE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "greedy": GREEDY
            }
            results.append(result)
            
            print(f"\nProbe: {probe}")
            print(f"Label: {label}")
            print(f"Response: {response[:200]}...")
            
            # Clean up
            del model
            torch.mps.empty_cache() if DEVICE == "mps" else torch.cuda.empty_cache()
    
    # Save results
    output_file = Path("multi_neuron_systematic_results.jsonl")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\nSUMMARY BY NEURON COUNT:")
    for count in NEURON_COUNTS:
        count_results = [r for r in results if r["neuron_count"] == count]
        label_counts = {}
        for r in count_results:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        print(f"\n{count} neurons: {label_counts}")

if __name__ == "__main__":
    run_experiment()
