import torch
from functools import partial
from dataclasses import dataclass
from transformer_lens.HookedTransformer import HookedTransformer

@dataclass
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: object
    validation_data: torch.Tensor
    validation_labels: torch.Tensor
    validation_mask: object
    validation_patch_data: torch.Tensor
    test_metrics: dict
    test_data: torch.Tensor
    test_labels: torch.Tensor
    test_mask: object
    test_patch_data: torch.Tensor

from acdc.ioi.ioi_dataset import IOIDataset
import torch.nn.functional as F
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs

def get_pythia_model(model_name="EleutherAI/pythia-410m", device="cuda"):
    tl_model = HookedTransformer.from_pretrained(model_name)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_ioi_data(tl_model, num_examples=100, device="cuda", metric_name="kl_div"):
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
        nb_templates=1,
        seed=0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    seq_len = ioi_dataset.toks.shape[1]
    
    default_data = ioi_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[:num_examples*2, seq_len-1]
    wrong_labels = torch.as_tensor(ioi_dataset.s_tokenIDs[:num_examples*2], dtype=torch.long, device=device)

    labels = labels.to(device)

    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]

    with torch.no_grad():
        base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=True,
        )
    elif metric_name == "logit_diff":
        validation_metric = partial(
            logit_diff_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    test_metrics = {}

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )
