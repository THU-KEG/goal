'''
    In the official code https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py,
    according to the main function `from_pretrained` and its sub-called functions `_load_pretrained_model` and `_load_state_dict_into_model`.
'''
import logging
from typing import List
import re

logger = logging.getLogger(__name__)




def init_missing_modules(model, loaded_state_dict):
    '''
    '''
    # Retrieve missing & unexpected_keys
    loaded_keys = [key for key in loaded_state_dict.keys()]
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    prefix = model.base_model_prefix

    def _fix_key(key):
        if "beta" in key:
            return key.replace("beta", "bias")
        if "gamma" in key:
            return key.replace("gamma", "weight")
        return key

    original_loaded_keys = loaded_keys
    loaded_keys = [_fix_key(key) for key in loaded_keys]

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
        expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if model._keys_to_ignore_on_load_missing is not None:
        for pat in model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    # retrieve unintialized modules and initialize before maybe overriding that with the pretrained weights.
    uninitialized_modules = model.retrieve_modules_from_names(
        missing_keys, add_prefix=add_prefix_to_model, remove_prefix=remove_prefix_from_model
    )
    for module in uninitialized_modules:
        model._init_weights(module)

    return model

def load_weight(model, state_dict, freeze_pretrained=True, fast_init=True):
    ''' Mostly inhirent from the function `_load_state_dict_into_model`.
    '''
    if fast_init:
        model = init_missing_modules(model, state_dict)

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")

    # Freeze the pretrained parameters if needed
    if freeze_pretrained:
        loaded_keys = state_dict.keys()
        for n,p in start_model.named_parameters():
            if n in loaded_keys:
                p.requires_grad = False

    # Make sure we are still sharing the output and input embeddings after loading weights
    model.set_tied()
    # model.tie_weights()

    return model