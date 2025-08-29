# Meteor: https://meteorfrom.space/

Meteor is a provably-secure symmetric-key steganography algorithm that encode & decodes messages with an LLM (GPT-2). Its encoded messages are statistically indistinguishable from regular LLM outputs.

```console
# Encode our message
$ docker run ghcr.io/rohanssrao/meteor:0.0.1 encode \
  --message "test message" \
  --password "password123"
 In the background, there is the apparent question of how to present to a user a wide variety of expressive values, but what scenario should

# Decode to get the message back
$ docker run ghcr.io/rohanssrao/meteor:0.0.1 decode \
  --message " In the background, there is the apparent question of how to present to a user a wide variety of expressive values, but what scenario should" \
  --password "password123"
test message
```

