import torch
import os
import json
import argparse
from transformers import PreTrainedModel, PretrainedConfig, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutput
import lm_eval
from lm_eval.models.huggingface import HFLM

from src.gpt import GPT
from src.gpt_config import GPTConfig

class CustomConfig(PretrainedConfig):
    model_type = "custom_gpt"

class CustomHFWrapper(PreTrainedModel):
    config_class = CustomConfig
    
    def __init__(self, config, custom_model):
        super().__init__(config)
        self.model = custom_model
        
    def forward(self, input_ids, **kwargs):
        logits, _ = self.model(input_ids, targets=None)
        return CausalLMOutput(logits=logits)
        
    @property
    def device(self):
        return next(self.model.parameters()).device

def run_comparison(ckpt_path, out_dir, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "-"*50)
    print("STEP 1: Evaluating OpenAI's Original GPT-2 (124M)")
    print("-"*50)
    
    baseline_lm = HFLM(pretrained="gpt2", backend="causal", batch_size=batch_size)
    baseline_results = lm_eval.simple_evaluate(
        model=baseline_lm,
        tasks=["hellaswag"],
        device=device,
        batch_size=batch_size
    )
    base_acc = baseline_results['results']['hellaswag']['acc,none']
    base_acc_norm = baseline_results['results']['hellaswag']['acc_norm,none']

    # Free up VRAM
    del baseline_lm
    torch.cuda.empty_cache()

    print("\n" + "-"*50)
    print("STEP 2: Evaluating Your Custom Model")
    print("-"*50)
    
    config = GPTConfig(vocab_size=50304)
    my_model = GPT(config)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']

    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '').replace('module.', '')
        unwrapped_state_dict[new_key] = v

    my_model.load_state_dict(unwrapped_state_dict)
    my_model.to(device)
    my_model.eval()

    hf_config = CustomConfig()
    wrapped_model = CustomHFWrapper(hf_config, my_model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    custom_lm = HFLM(pretrained=wrapped_model, backend="causal", tokenizer=tokenizer, batch_size=batch_size)
    
    custom_results = lm_eval.simple_evaluate(
        model=custom_lm,
        tasks=["hellaswag"],
        device=device,
        batch_size=batch_size
    )
    custom_acc = custom_results['results']['hellaswag']['acc,none']
    custom_acc_norm = custom_results['results']['hellaswag']['acc_norm,none']


    report = f"""========================================
 HELLASWAG COMPARISON REPORT 
========================================
Dataset: HellaSwag (Common Sense Reasoning)

[ OpenAI's GPT-2 (124M Baseline) ]
Accuracy:           {base_acc * 100:.2f}%
Normalized Acc:     {base_acc_norm * 100:.2f}%

[ Your Custom Model ]
Accuracy:           {custom_acc * 100:.2f}%
Normalized Acc:     {custom_acc_norm * 100:.2f}%

[ Difference (Custom vs OpenAI) ]
Accuracy Diff:      {(custom_acc - base_acc) * 100:+.2f}%
Norm Acc Diff:      {(custom_acc_norm - base_acc_norm) * 100:+.2f}%
========================================"""

    print("\n" + report)

    report_path = os.path.join(out_dir, "hellaswag_comparison.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    json_data = {
        "openai_gpt2_baseline": {
            "accuracy": base_acc,
            "normalized_accuracy": base_acc_norm
        },
        "custom_model": {
            "accuracy": custom_acc,
            "normalized_accuracy": custom_acc_norm
        },
        "deltas": {
            "accuracy_diff": custom_acc - base_acc,
            "normalized_accuracy_diff": custom_acc_norm - base_acc_norm
        }
    }
    json_path = os.path.join(out_dir, "hellaswag_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"\nResults successfully saved to: {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Custom GPT vs OpenAI GPT-2 on HellaSwag")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to your .pt checkpoint")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Directory to save the comparison results")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    run_comparison(args.ckpt, args.out_dir, args.batch_size)