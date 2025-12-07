# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.51.0",
#     "accelerate",
#     "safetensors",
#     "numpy",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cpu"]
# ///

"""Meteor - Cryptographically secure steganography for realistic distributions.

A project by Gabe Kaptchuk, Tushar Jois, Matthew Green, and Avi Rubin.
Paper: https://eprint.iacr.org/2021/686

Usage:
    uv run meteor.py encode --message "secret" --password "pass"
    uv run meteor.py decode --message "stegotext" --password "pass"
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib
import hmac
import math
import os
import argparse
import sys


class TokenizerWrapper:
    """Wrapper to provide GPT-2 style .encoder/.decoder interface for modern tokenizers."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # Build encoder dict (token string -> id)
        self.encoder = tokenizer.get_vocab()
        # Build decoder dict (id -> token string)
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, int):
            return self.decoder.get(token_ids, '')
        return [self.decoder.get(tid, '') for tid in token_ids]

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)

class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0
        # Pre-compute bit extraction for faster generation
        self._bit_buffer = None
        self._buffer_pos = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

        # Reset buffer on reseed
        self._bit_buffer = None
        self._buffer_pos = 0

    def _refill_buffer(self):
        """Convert val bytes to bits efficiently using numpy."""
        # Convert first 8 bytes to bits (64 bits)
        val_bytes = np.frombuffer(self.val[:8], dtype=np.uint8)
        # Unpack each byte into 8 bits
        self._bit_buffer = np.unpackbits(val_bytes)
        self._buffer_pos = 0

    def generate_bits(self, n):
        """Generate n random bits efficiently."""
        xs = np.zeros(n, dtype=bool)
        pos = 0

        while pos < n:
            # Refill buffer if needed
            if self._bit_buffer is None or self._buffer_pos >= len(self._bit_buffer):
                self._refill_buffer()
                # Generate new val for next refill
                self.val = self.hmac(self.key, self.val)

            # Copy as many bits as we can from buffer
            available = len(self._bit_buffer) - self._buffer_pos
            needed = n - pos
            take = min(available, needed)

            xs[pos:pos+take] = self._bit_buffer[self._buffer_pos:self._buffer_pos+take]
            self._buffer_pos += take
            pos += take

        self.reseed()
        return xs



# Maximum context length for KV cache (Qwen3 supports 40960, but we limit for memory)
MAX_CONTEXT_LENGTH = 8190

def limit_past(past):
    """Limit past key-value cache to prevent memory issues.

    Modern transformers uses DynamicCache objects. We crop if seq_length exceeds limit.
    """
    if past is None:
        return None

    # Check if this is a DynamicCache or similar object
    if hasattr(past, 'get_seq_length'):
        seq_length = past.get_seq_length()
        if seq_length > MAX_CONTEXT_LENGTH:
            # Crop the cache - DynamicCache stores key_cache and value_cache lists
            if hasattr(past, 'key_cache') and hasattr(past, 'value_cache'):
                for layer_idx in range(len(past.key_cache)):
                    past.key_cache[layer_idx] = past.key_cache[layer_idx][:, :, -MAX_CONTEXT_LENGTH:, :]
                    past.value_cache[layer_idx] = past.value_cache[layer_idx][:, :, -MAX_CONTEXT_LENGTH:, :]
        return past

    # Fallback for tuple-based past (older format)
    new_past = []
    for layer_past in past:
        key, value = layer_past
        new_key = key[:, :, -MAX_CONTEXT_LENGTH:, :]
        new_value = value[:, :, -MAX_CONTEXT_LENGTH:, :]
        new_past.append((new_key, new_value))

    return tuple(new_past)

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i

# Qwen3 special token IDs
ENDOFTEXT_TOKEN_ID = 151643  # <|endoftext|>
IM_END_TOKEN_ID = 151645     # <|im_end|>

# Large negative value for blocking tokens (use -1e4 for float16 compatibility)
BLOCK_VALUE = -1e4

# Global device tracking for precision handling
_CURRENT_DEVICE = 'cpu'

def to_high_precision(tensor):
    """Convert tensor to highest precision available on current device."""
    if _CURRENT_DEVICE == 'mps':
        return tensor.float()  # MPS doesn't support float64
    return tensor.double()

def encode_context(raw_text, enc):
    context_tokens = [ENDOFTEXT_TOKEN_ID] + enc.encode(raw_text)
    return context_tokens

MODEL_NAME = 'Qwen/Qwen3-0.6B'

def is_model_cached(model_name):
    """Check if a model is already downloaded in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if config.json is cached (indicates model is downloaded)
        result = try_to_load_from_cache(model_name, "config.json")
        return result is not None
    except Exception:
        return True  # Assume cached if we can't check

def get_model(seed=1234):
    """Load the model from HuggingFace with automatic device selection."""
    global _CURRENT_DEVICE, device

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Only show download message if model isn't cached
    if not is_model_cached(MODEL_NAME):
        print(f"Downloading model {MODEL_NAME}...", file=sys.stderr, flush=True)

    # Load tokenizer and wrap for GPT-2 style interface
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    enc = TokenizerWrapper(tokenizer)

    # Load model with automatic device placement
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    # Set global device for tensor placement and precision handling
    device = model.device
    _CURRENT_DEVICE = 'mps' if 'mps' in str(device) else str(device).split(':')[0]

    return enc, model

def bin_sort(l, token_indices, total, entropy, device):
    #compute entropy for upper bound on the number of bins we need

    bucket_size = total
    num_bins = 2**int(entropy+1)
    bucket_size = total / num_bins

    bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins
    value_in_bins = [0] * num_bins
    space_left_after = [total - i*bucket_size for i in range(0,num_bins)]


    token_bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins

    # Figuring out what the search order should be
    step_size = num_bins/4
    search_order = []
    priorities = [0]*num_bins
    priority = 0
    search_order.append(int(num_bins/2))
    search_order.append(0)
    priorities[int(num_bins/2)] = 0
    priorities[0] = 0
    while(step_size>=1):
        priority += 1
        for x in range(num_bins-int(step_size), -1, -int(step_size*2)):
            search_order.append(x)
            priorities[x] = priority
        step_size = step_size/2

    # Adding the actual elements
    for (item, token_index) in zip(l.tolist(), token_indices.tolist()):
        found_single_bucket_fit = False
        single_bucket_index = -1
        single_bucket_value = bucket_size

        found_multi_bucket_bumpless_fit = False
        multi_bucket_bumpless_index = -1
        multi_bucket_bumpless_value = total

        found_multi_bucket_bumping_fit = False
        multi_bucket_bumping_index = -1
        multi_bucket_bumping_value = total

        for i in search_order:  # for index in search_order
            if(item > space_left_after[i]):
                continue
            if(value_in_bins[i] >= bucket_size):
                continue

            # Priority of choices
            #  1. Can i place this thing in an empty bucket all on its own?
            #  2. Can i plan this somewhere where is doesnt have to bump anything else around?
            #    2a. Minimize the wasted space.  Aka use the smallest space (of equal priority) that accomplishes this goal
            #  3. If not (1) and (2), then put it in the space the bumps stuff the least.

            if(value_in_bins[i] + item > bucket_size): #Would overflow.

                space_before_next_block = bucket_size - value_in_bins[i]
                for j in range(i+1, len(bins)):
                    if(value_in_bins[j] > 0): # We have found a bucket with something in it.  This is how much space we have here.
                        space_before_next_block = space_before_next_block + (bucket_size - value_in_bins[i])
                        break
                    else: # This was a empty bucket
                        space_before_next_block = space_before_next_block + bucket_size

                if((not found_multi_bucket_bumpless_fit) or (found_multi_bucket_bumpless_fit and priorities[i] <= priorities[multi_bucket_bumpless_index])): #This could potentially be a match

                    # If this is a valid space to put this without bumping and it is a better fit than previous spaces
                    if(space_before_next_block > item and space_before_next_block < multi_bucket_bumpless_value):
                        # set this to be the pointer!  we can fit stuff here
                        found_multi_bucket_bumpless_fit = True
                        multi_bucket_bumpless_index = i
                        multi_bucket_bumpless_value = space_before_next_block

                    # Find the overflow that will bump the least
                    if ( item - space_before_next_block < multi_bucket_bumping_value):
                        found_multi_bucket_bumping_fit = True
                        multi_bucket_bumping_index = i
                        multi_bucket_bumping_value = item - space_before_next_block

            if(value_in_bins[i] + item <= bucket_size): #Would fit
                if(single_bucket_value > value_in_bins[i]):
                    found_single_bucket_fit = True
                    single_bucket_value = value_in_bins[i]
                    single_bucket_index = i

        if (single_bucket_index == multi_bucket_bumpless_index == multi_bucket_bumping_index == -1):
            bins[0] = torch.cat( (torch.tensor([item], device=device), bins[0]), 0)
            token_bins[0] = torch.cat( (torch.tensor([token_index], device=device), token_bins[0]), 0)
            continue


        if found_single_bucket_fit:
            # We found somewhere we can actually fit!
            bins[single_bucket_index] = torch.cat( (bins[single_bucket_index], torch.tensor([item], device=device)), 0)
            token_bins[single_bucket_index] = torch.cat( (token_bins[single_bucket_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[single_bucket_index] += item
            for i in range(0, single_bucket_index+1):
                space_left_after[i] -= item

        elif found_multi_bucket_bumpless_fit:
            # Found somewhere we can put this without upsetting the force
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumpless_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumpless_index] = torch.cat( (bins[multi_bucket_bumpless_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumpless_index] = torch.cat( (token_bins[multi_bucket_bumpless_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumpless_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumpless_index + 1
            for i in range(0, j):
                space_left_after[i] -= item

            while(part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow+value_in_bins[j]) # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j+=1

        else:
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumping_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumping_index] = torch.cat( (bins[multi_bucket_bumping_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumping_index] = torch.cat( (token_bins[multi_bucket_bumping_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumping_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumping_index + 1
            for i in range(0, j):
                space_left_after[i] -= item
            while(part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow+value_in_bins[j]) # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j+=1



    sorted_tensor = torch.cat(bins, 0)
    sorted_tokens = torch.cat(token_bins, 0)

    return sorted_tensor, sorted_tokens

# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_key = b'0x01'*64
sample_seed_prefix = b'sample'
sample_nonce_counter = b'\x00' * 16

def encode_meteor(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=50000, is_sort=False, randomize_key=False, input_key=sample_key, input_nonce=sample_nonce_counter):

    if randomize_key:
        input_key = os.urandom(64)
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)
    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)


    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    # Pre-allocate output list for efficiency (avoid repeated torch.cat)
    output_tokens = []
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0

    # Use inference_mode for slightly better performance than no_grad
    with torch.inference_mode():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            # Block special tokens from being generated
            logits[0, -1, ENDOFTEXT_TOKEN_ID] = BLOCK_VALUE  # <|endoftext|>
            logits[0, -1, IM_END_TOKEN_ID] = BLOCK_VALUE     # <|im_end|>
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = to_high_precision(logits)
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k] # Cutoff all but top k
                indices = indices[:k]

                # Rescale to correct range
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()

                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution, device)
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Apply the mask to the message
                message_bits = message[i:i+precision]
                if i+precision > len(message):
                    message_bits = message_bits + [0]*(i+precision-len(message))

                mask_bits = mask_generator.generate_bits(precision)
                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = to_high_precision(probs_final)/probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy_in_this_distribution
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output_tokens.append(prev.item())
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output_tokens)
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    return output_tokens, avg_NLL, avg_KL, words_per_bit, avg_Hq

def decode_meteor(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=50000, is_sort=False, input_key=sample_key, input_nonce=sample_nonce_counter):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)

    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)

    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.inference_mode():
        i = 0
        while i < len(inp):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            # Block special tokens
            logits[0, -1, ENDOFTEXT_TOKEN_ID] = BLOCK_VALUE  # <|endoftext|>
            logits[0, -1, IM_END_TOKEN_ID] = BLOCK_VALUE     # <|im_end|>
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = to_high_precision(logits)
            logits_temp = logits / temp
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1]-cur_interval[0]
            cur_threshold = 1/cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k] # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range
            entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution, device)
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle BPE tokenization mismatches
            if rank >= k:
                true_token_text = enc.decoder.get(inp[i], '')
                for rank_idx in range(k):
                    prop_token_text = enc.decoder.get(indices[rank_idx].item(), '')

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix) # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                              true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder.get(inp[i+num_extra], '')
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i+j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix) # a list
                                inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text), file=sys.stderr)
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp)-1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            # Get the mask and apply it to the recovered bits
            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)

            i += 1

    return message

def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=50000):

    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    output_tokens = []  # Use list for efficiency
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0

    with torch.inference_mode():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            # Block special tokens
            logits[0, -1, ENDOFTEXT_TOKEN_ID] = BLOCK_VALUE  # <|endoftext|>
            logits[0, -1, IM_END_TOKEN_ID] = BLOCK_VALUE     # <|im_end|>
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = to_high_precision(logits)
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k] # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i+precision]
                if i+precision > len(message):
                    message_bits = message_bits + [0]*(i+precision-len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = to_high_precision(probs_final)/probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output_tokens.append(prev.item())
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output_tokens)
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    return output_tokens, avg_NLL, avg_KL, words_per_bit, avg_Hq

def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=50000):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)

    context = torch.tensor(context[-MAX_CONTEXT_LENGTH:], device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.inference_mode():
        i = 0
        while i < len(inp):
            outputs = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = limit_past(outputs.past_key_values)
            # Block special tokens
            logits[0, -1, ENDOFTEXT_TOKEN_ID] = BLOCK_VALUE  # <|endoftext|>
            logits[0, -1, IM_END_TOKEN_ID] = BLOCK_VALUE     # <|im_end|>
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = to_high_precision(logits)
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1]-cur_interval[0]
            cur_threshold = 1/cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k] # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle BPE tokenization mismatches
            if rank >= k:
                true_token_text = enc.decoder.get(inp[i], '')
                for rank_idx in range(k):
                    prop_token_text = enc.decoder.get(indices[rank_idx].item(), '')

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix) # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                              true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder.get(inp[i+num_extra], '')
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i+j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix) # a list
                                inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text), file=sys.stderr)
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp)-1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return message

def bytes_to_bits(data):
    """Convert bytes to list of bits."""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def bits_to_bytes(bits):
    """Convert list of bits back to bytes."""
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)

    result = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return bytes(result)

def encode_message(message_str, context, key, nonce):
    finish_sent = False
    meteor_sort = False
    meteor_random = False

    context_tokens = encode_context(context, enc)

    # Convert message to raw UTF-8 bytes, then to bits
    # Prefix with length (4 bytes) so we know where message ends
    message_bytes = message_str.encode('utf-8')
    length_bytes = len(message_bytes).to_bytes(4, 'big')
    message = bytes_to_bits(length_bytes + message_bytes)

    # Encode bits into cover text using Meteor
    Hq = 0
    out, nll, kl, words_per_bit, Hq = encode_meteor(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent,
                                                    precision=precision, topk=topk, device=device, is_sort=meteor_sort, randomize_key=meteor_random, input_key=key, input_nonce=nonce)
    text = enc.decode(out)

    stats = {
        "ppl": math.exp(nll),
        "kl": kl,
        "wordsbit": words_per_bit,
        "entropy": Hq/0.69315
    }
    return text, stats

def decode_message(text, context, key, nonce):
    meteor_sort = False

    context_tokens = encode_context(context, enc)

    # Decode bits from stegotext using Meteor
    message_bits = decode_meteor(model, enc, text, context_tokens, temp=temp,
                                precision=precision, topk=topk, device=device, is_sort=meteor_sort, input_key=key, input_nonce=nonce)

    # Convert bits back to bytes
    message_bytes = bits_to_bytes(message_bits)

    # Extract length prefix and message
    if len(message_bytes) < 4:
        return ""
    length = int.from_bytes(message_bytes[:4], 'big')
    message_bytes = message_bytes[4:4+length]

    return message_bytes.decode('utf-8', errors='replace')

"""## Contexts

A context is the initial state of the GPT-2 algorithm, prior to running any sampling. This impacts the topic of the output of the model. Use the following form field, `chosen-context`, to select one. We have provided three example contexts:

* "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."

* "The Alvarez hypothesis posits that the mass extinction of the dinosaurs and many other living things during the Cretaceous-Paleogene extinction event was caused by the impact of a large asteroid on the Earth. Prior to 2013, it was commonly cited as having happened about 65 million years ago, but Renne and colleagues (2013) gave an updated value of 66 million years. Evidence indicates that the asteroid fell in the Yucatan Peninsula, at Chicxulub, Mexico. The hypothesis is named after the father-and-son team of scientists Luis and Walter Alvarez, who first suggested it in 1980. Shortly afterwards, and independently, the same was suggested by Dutch paleontologist Jan Smit."

* "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist."

Feel free to edit and add your own!
"""

parser=argparse.ArgumentParser(
    description="Meteor: Cryptographically secure steganography using Qwen3",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python meteor.py encode --message "secret" --password "pass123"
  python meteor.py decode --message "stegotext here" --password "pass123"
  python meteor.py encode --message "secret" --password "pass" --context "Custom context..."
"""
)

default_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist."

parser.add_argument("mode", choices=["encode", "decode"], help="mode: encode or decode", default="encode")
parser.add_argument("--message", help="secret message to encode/decode", required=True)
parser.add_argument("--context", help="prior context for the LLM", default=default_context)
parser.add_argument("--password", help="password to encode with", required=True)
args=parser.parse_args()

message = args.message
context = args.context

password = args.password.encode('utf-8')
salt = b'salt_'
key = hashlib.pbkdf2_hmac('sha256', password, salt, 100000, dklen=64)
nonce = b'\x01'*64

temp = 0.95
precision = 32
topk = 50000

enc, model = get_model()

if args.mode == "encode":
    print(encode_message(message, context, key, nonce)[0])
else:
    print(decode_message(message, context, key, nonce))
