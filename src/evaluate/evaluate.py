import torch
import os
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

# --- 2. The Evaluation Logic ---

def evaluate_model(ckpt_path, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing custom GPT architecture...")
    config = GPTConfig(vocab_size=50304)
    my_model = GPT(config)

    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']

    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '').replace('module.', '')
        unwrapped_state_dict[new_key] = v

    my_model.load_state_dict(unwrapped_state_dict)
    my_model.to(device)
    my_model.eval()

    print("Wrapping model for lm-eval...")
    hf_config = CustomConfig()
    wrapped_model = CustomHFWrapper(hf_config, my_model)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    lm_obj = HFLM(pretrained=wrapped_model, backend="causal", tokenizer=tokenizer, batch_size=batch_size)

    print("Starting HellaSwag evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["hellaswag"],
        device=device,
        batch_size=batch_size
    )

    accuracy = results['results']['hellaswag']['acc,none']
    acc_norm = results['results']['hellaswag']['acc_norm,none']
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Hellaswag Accuracy:          {accuracy * 100:.2f}%")
    print(f"Hellaswag Normalized Acc:    {acc_norm * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Custom GPT on HellaSwag")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to your .pt checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    evaluate_model(args.ckpt, args.batch_size)