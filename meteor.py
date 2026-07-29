# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=5.2.0",
#     "accelerate",
#     "safetensors",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cpu"]
# ///

"""Meteor: language-model steganography using LFM2.5-230M-Base.

Based on the Meteor paper by Gabe Kaptchuk, Tushar Jois, Matthew Green,
and Avi Rubin: https://eprint.iacr.org/2021/686
"""

import argparse
import hashlib
import hmac
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LiquidAI/LFM2.5-230M-Base"
MODEL_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"
TEMPERATURE = 0.95
TOP_K = 200
PRECISION = 32
PROBABILITY_RANGE = 1 << PRECISION
MAX_CONTEXT_TOKENS = 8190
MAX_FINISH_TOKENS = 128
MAX_STALLED_TOKENS = 256
KDF_SALT = b"salt_"
KDF_ITERATIONS = 100_000
STREAM_DOMAIN = b"meteor-payload-v1"

DEFAULT_CONTEXT = (
    "Despite a long history of research and wide-spread applications to censorship "
    "resistant systems, practical steganographic systems capable of embedding messages "
    "into realistic communication distributions, like text, do not exist."
)


@dataclass
class Runtime:
    tokenizer: object
    model: object
    device: torch.device
    blocked_token_ids: tuple[int, ...]


@dataclass
class Distribution:
    token_ids: torch.Tensor
    cumulative: torch.Tensor


class HmacBitstream:
    """Deterministic pseudorandom bitstream used to mask payload bits."""

    def __init__(self, key: bytes, domain: bytes = STREAM_DOMAIN):
        self.key = key
        self.domain = domain
        self.counter = 0
        self.buffer: list[int] = []

    def generate(self, count: int) -> list[int]:
        while len(self.buffer) < count:
            block = hmac.new(
                self.key,
                self.domain + self.counter.to_bytes(8, "big"),
                hashlib.sha512,
            ).digest()
            self.counter += 1
            self.buffer.extend(bytes_to_bits(block))

        result = self.buffer[:count]
        del self.buffer[:count]
        return result


def is_model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(model_name, "config.json") is not None
    except Exception:
        return True


def load_runtime(seed: int = 1234) -> Runtime:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not is_model_cached(MODEL_NAME):
        print(f"Downloading model {MODEL_NAME}...", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    # LFM exposes hundreds of control tokens in its vocabulary while marking
    # only a few of them as special. None belong in visible cover text.
    control_ids = {
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token.startswith("<|") and token.endswith("|>")
    }
    blocked_ids = tuple(sorted(set(tokenizer.all_special_ids) | control_ids))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return Runtime(tokenizer, model, model.device, blocked_ids)


def derive_key(password: str) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        KDF_SALT,
        KDF_ITERATIONS,
        dklen=64,
    )


def normalize_context(context: str) -> str:
    """Add a word boundary so generated text does not begin with a space."""
    if context and not context[-1].isspace():
        return context + " "
    return context


def encode_context(context: str, runtime: Runtime) -> list[int]:
    tokenizer = runtime.tokenizer
    start_id = tokenizer.bos_token_id
    if start_id is None:
        start_id = tokenizer.all_special_ids[0]
    token_ids = [start_id] + tokenizer.encode(
        normalize_context(context),
        add_special_tokens=False,
    )
    return token_ids[-MAX_CONTEXT_TOKENS:]


def crop_cache(cache):
    if cache is None or not hasattr(cache, "get_seq_length"):
        return cache
    if cache.get_seq_length() <= MAX_CONTEXT_TOKENS:
        return cache

    if hasattr(cache, "crop"):
        cache.crop(MAX_CONTEXT_TOKENS)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer] = cache.key_cache[layer][
                :, :, -MAX_CONTEXT_TOKENS:, :
            ]
            cache.value_cache[layer] = cache.value_cache[layer][
                :, :, -MAX_CONTEXT_TOKENS:, :
            ]
    return cache


def contains_unsafe_whitespace(text: str) -> bool:
    """Allow ASCII spaces but reject line breaks, tabs, and Unicode whitespace."""
    return any(character != " " and character.isspace() for character in text)


def canonical_suffix(prefix: list[int], runtime: Runtime) -> list[int]:
    """Return the final pre-token, the only part changed by appending text."""
    if not prefix:
        return []

    tokenizer = runtime.tokenizer
    text = tokenizer.decode(
        prefix,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    pieces = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    if not pieces:
        return prefix

    final_piece_start = pieces[-1][1][0]
    stable_ids = tokenizer(
        text[:final_piece_start],
        add_special_tokens=False,
    )["input_ids"]

    # Unexpected offsets only cost performance: checking the full prefix is safe.
    if prefix[: len(stable_ids)] != stable_ids:
        return prefix
    return prefix[len(stable_ids) :]


def allowed_candidate_positions(
    prefix: list[int], candidate_ids: list[int], runtime: Runtime
) -> list[int]:
    """Find candidate tokens that survive text round-tripping on one line."""
    tokenizer = runtime.tokenizer
    suffix = canonical_suffix(prefix, runtime)
    sequences = [suffix + [token_id] for token_id in candidate_ids]
    texts = tokenizer.batch_decode(
        sequences,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    retokenized = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    positions = []
    for position, (text, expected, actual) in enumerate(
        zip(texts, sequences, retokenized)
    ):
        if expected != actual:
            continue
        if contains_unsafe_whitespace(text):
            continue
        if not prefix and text.startswith(" "):
            continue
        positions.append(position)

    if len(positions) < 2:
        raise RuntimeError("Model produced fewer than two safe next-token choices")
    return positions


def high_precision(tensor: torch.Tensor) -> torch.Tensor:
    # MPS has no float64 support. Model logits are bfloat16, so float32 retains
    # all their information; other devices use float64 for stable interval math.
    return tensor.float() if tensor.device.type == "mps" else tensor.double()


def build_distribution(
    logits: torch.Tensor,
    prefix: list[int],
    runtime: Runtime,
) -> Distribution:
    logits = logits.clone()
    logits[list(runtime.blocked_token_ids)] = -1e4
    values, token_ids = logits.topk(TOP_K, sorted=True)
    candidate_ids = token_ids.tolist()  # one accelerator synchronization
    positions = allowed_candidate_positions(prefix, candidate_ids, runtime)
    position_tensor = torch.tensor(positions, device=runtime.device)

    token_ids = token_ids[position_tensor]
    values = high_precision(values[position_tensor]) / TEMPERATURE
    probabilities = F.softmax(values, dim=0)

    # Allocate every integer point exactly once. Flooring can leave at most one
    # point per candidate; assigning the remainder to the top token is negligible.
    masses = torch.floor(probabilities * PROBABILITY_RANGE).long()
    remainder = PROBABILITY_RANGE - int(masses.sum().item())
    masses[0] += remainder
    cumulative = masses.cumsum(0)
    return Distribution(token_ids, cumulative)


def bytes_to_bits(data: bytes) -> list[int]:
    return [
        (byte >> shift) & 1
        for byte in data
        for shift in range(7, -1, -1)
    ]


def bits_to_bytes(bits: list[int]) -> bytes:
    complete_length = len(bits) - (len(bits) % 8)
    return bytes(
        bits_to_int(bits[offset : offset + 8])
        for offset in range(0, complete_length, 8)
    )


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def int_to_bits(value: int, width: int = PRECISION) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def common_prefix_length(first: list[int], second: list[int]) -> int:
    for position, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return position
    return len(first)


def interval_bits(distribution: Distribution, selection: int):
    low = int(distribution.cumulative[selection - 1].item()) if selection else 0
    high = int(distribution.cumulative[selection].item())
    low_bits = int_to_bits(low)
    high_bits = int_to_bits(high - 1)
    fixed = common_prefix_length(low_bits, high_bits)
    return low_bits, high_bits, fixed


def text_is_finished(token_ids: list[int], runtime: Runtime) -> bool:
    text = runtime.tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).rstrip().rstrip('"\'”’)]}')
    if text.endswith(("!", "?")):
        return True
    if not text.endswith("."):
        return False
    final_word = text[:-1].rstrip().rsplit(maxsplit=1)[-1].strip('"\'“‘([{')
    return len(final_word) > 1


def encode_cover(
    secret: str,
    context: str,
    key: bytes,
    runtime: Runtime,
) -> str:
    payload = secret.encode("utf-8")
    if len(payload) >= 1 << 32:
        raise ValueError("Message is too large")
    payload_bits = bytes_to_bits(len(payload).to_bytes(4, "big") + payload)

    stream = HmacBitstream(key)
    context_ids = encode_context(context, runtime)
    previous = torch.tensor(context_ids, device=runtime.device, dtype=torch.long)
    cache = None
    output_ids: list[int] = []
    bit_offset = 0
    stalled_tokens = 0
    finish_tokens = 0

    with torch.inference_mode():
        while True:
            result = runtime.model(
                previous.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            cache = crop_cache(result.past_key_values)
            distribution = build_distribution(
                result.logits[0, -1, :],
                output_ids,
                runtime,
            )

            if bit_offset < len(payload_bits):
                chunk = payload_bits[bit_offset : bit_offset + PRECISION]
                chunk += [0] * (PRECISION - len(chunk))
                mask = stream.generate(PRECISION)
                point = bits_to_int(a ^ b for a, b in zip(chunk, mask))
                selection = int(
                    (distribution.cumulative > point).nonzero()[0].item()
                )
                _, _, consumed = interval_bits(distribution, selection)
                bit_offset += consumed
                stalled_tokens = stalled_tokens + 1 if consumed == 0 else 0
                if stalled_tokens >= MAX_STALLED_TOKENS:
                    raise RuntimeError(
                        "Generation stalled on a distribution with insufficient entropy"
                    )
            else:
                selection = 0
                finish_tokens += 1

            token_id = int(distribution.token_ids[selection].item())
            output_ids.append(token_id)
            previous = torch.tensor(
                [token_id],
                device=runtime.device,
                dtype=torch.long,
            )

            if bit_offset >= len(payload_bits) and (
                text_is_finished(output_ids, runtime)
                or finish_tokens >= MAX_FINISH_TOKENS
            ):
                break

    cover = runtime.tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if contains_unsafe_whitespace(cover):
        raise RuntimeError("Internal error: generated cover contains unsafe whitespace")
    return cover


def decode_cover(
    cover: str,
    context: str,
    key: bytes,
    runtime: Runtime,
) -> str:
    if contains_unsafe_whitespace(cover):
        raise ValueError("Stegotext must contain only ASCII spaces as whitespace")

    input_ids = runtime.tokenizer.encode(cover, add_special_tokens=False)
    context_ids = encode_context(context, runtime)
    previous = torch.tensor(context_ids, device=runtime.device, dtype=torch.long)
    cache = None
    decoded_bits: list[int] = []
    stream = HmacBitstream(key)

    with torch.inference_mode():
        for position, token_id in enumerate(input_ids):
            result = runtime.model(
                previous.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            cache = crop_cache(result.past_key_values)
            distribution = build_distribution(
                result.logits[0, -1, :],
                input_ids[:position],
                runtime,
            )

            matches = (distribution.token_ids == token_id).nonzero()
            if len(matches) != 1:
                raise ValueError(
                    "Stegotext does not match this context, model, or Meteor version "
                    f"at token {position}"
                )
            selection = int(matches[0].item())
            _, high_bits, fixed = interval_bits(distribution, selection)
            mask = stream.generate(PRECISION)
            decoded_bits.extend(
                bit ^ mask[index]
                for index, bit in enumerate(high_bits[:fixed])
            )

            if len(decoded_bits) >= 32:
                payload_length = bits_to_int(decoded_bits[:32])
                target_bits = 32 + payload_length * 8
                if len(decoded_bits) >= target_bits:
                    payload = bits_to_bytes(decoded_bits[32:target_bits])
                    try:
                        return payload.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            "Unable to decode payload; check the password and stegotext"
                        ) from error

            previous = torch.tensor(
                [token_id],
                device=runtime.device,
                dtype=torch.long,
            )

    raise ValueError(
        "Stegotext ended before a complete payload was recovered; "
        "check the password, context, and stegotext"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Meteor: cryptographically secure language-model steganography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python meteor.py encode --message "secret" --password "pass123"
  python meteor.py decode --message "stegotext here" --password "pass123"
  python meteor.py encode --message "secret" --password "pass" --context "Custom context..."
""",
    )
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("--message", required=True, help="secret message or stegotext")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="prior model context")
    parser.add_argument("--password", required=True, help="password used for encoding")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime = load_runtime()
    key = derive_key(args.password)

    try:
        if args.mode == "encode":
            result = encode_cover(args.message, args.context, key, runtime)
        else:
            result = decode_cover(args.message, args.context, key, runtime)
    except (RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    print(result)


if __name__ == "__main__":
    main()
