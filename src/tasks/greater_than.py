import torch
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.TLACDCExperiment import TLACDCExperiment

def get_greaterthan_data(model, num_examples=20, device="cpu", metric_name="prob_diff"):
    """
    Sets up the Greater-Than task for the given HookedTransformer model.
    metric_name: "prob_diff" or "kl_div"
    """
    things = get_all_greaterthan_things(
        num_examples=num_examples,
        device=device,
        metric_name=metric_name
    )
    return things

