# Meteor

Cryptographically secure steganography using Qwen3-0.6B.

Meteor encodes secret messages into innocent-looking AI-generated text. The encoded messages are statistically indistinguishable from regular LLM outputs.

This is an unofficial implementation of the ideas in the [original Meteor paper](https://eprint.iacr.org/2021/686) by Kaptchuk, Jois, Green, and Rubin, adapted for a modern LLM. An explanation of Meteor can be found at this link: https://meteorfrom.space

## Usage

### With Docker

```bash
docker run ghcr.io/rohanssrao/meteor:1.0.0 encode --message "secret" --password "pass"
docker run ghcr.io/rohanssrao/meteor:1.0.0 decode --message "<stegotext>" --password "pass"
```

### With uv

```bash
uv run meteor.py encode --message "secret" --password "pass"
```

## Custom Context

You can control the initial context of the generated text with `--context`:

```bash
uv run meteor.py encode --message "secret" --password "pass" \
  --context "The history of ancient Rome"
```

**Important:** The context, password, and stegotext must all match exactly when decoding. If you use a custom context during encoding, you must provide the same context when decoding:

```bash
# Encode with custom context
uv run meteor.py encode --message "secret" --password "pass" \
  --context "The history of ancient Rome"

# Decode
uv run meteor.py decode --message "<stegotext>" --password "pass" \
  --context "The history of ancient Rome"
```

## How It Works

1. Your message is converted to bits
2. These bits guide token selection from the LLM's probability distribution
3. The selected tokens form natural-looking text
4. Decoding reverses the process using the same password and context
