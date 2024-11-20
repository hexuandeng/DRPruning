""" The file contains the util functions to convert the composer model to the huggingface model or vice versa. """

""" convert composer weights to hf weights and test the equivalence """
import sys
import torch
from omegaconf import OmegaConf as om
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from drpruning.utils.hf_to_composer import get_key_map_from_hf_to_composer, get_layer_num_from_weights
from drpruning.utils.rewrite_qwen import rewrite_qwen

def get_key_map_from_composer_to_hf(num_layers):
    """ get kepmap from composer to hf """
    return {value: key for key, value in get_key_map_from_hf_to_composer(num_layers).items()}

def construct_hf_config(model_config: om = None):
    assert model_config is not None, "model config is None"
    model_class = model_config.pop("model_class")
    
    if model_class == "Llama2":
        hf_model_name = "drpruning/models/Llama-2-7b-hf"
        tokenzier_name = "drpruning/models/Llama-2-7b-hf"
        config = AutoConfig.from_pretrained(hf_model_name)
    elif model_class == "Qwen2":
        hf_model_name = "drpruning/models/Qwen2-7B"
        tokenzier_name = "drpruning/models/Qwen2-7B"
        config = AutoConfig.from_pretrained(hf_model_name)

    for key in model_config:
        setattr(config, key, model_config[key])
        print(config)

    return config, tokenzier_name 

def save_composer_to_hf(composer_model_path, output_path=None, model_config:om = None):
    """ convert composer ckpt's weights to huggingface """

    weights = torch.load(composer_model_path)
    if "state" in weights:
        weights = weights["state"]["model"]
    num_layers = get_layer_num_from_weights(weights)
    keymap = get_key_map_from_composer_to_hf(num_layers)
    hf_weights = {keymap[key]: weights[key] for key in weights if "rotary" not in key}
    config, tokenizer_nanme = construct_hf_config(model_config)

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(hf_weights, strict=False)
    model = model.bfloat16()
    model.save_pretrained(output_path, dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_nanme)
    tokenizer.save_pretrained(output_path)
    
    print(f"saved hf model to {output_path}")
   
if __name__ == "__main__":
    composer_model_path, output_path, other_args = sys.argv[1], sys.argv[2], sys.argv[3:]
    cli_cfg = om.from_cli(other_args)
    save_composer_to_hf(composer_model_path, output_path, cli_cfg)