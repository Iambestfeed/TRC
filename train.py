import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_hub


gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset('my_model')
gemma_lm.summary()

template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

import json

prompts = []
responses = []
line_count = 0

with open("databricks-dolly-15k.jsonl") as file:
    for line in file:

        examples = json.loads(line)
        # Filter out examples with context, to keep it simple.
        if examples["context"]:
            continue
        # Format data into prompts and response lists.
        prompts.append(examples["instruction"])
        responses.append(examples["response"])

        line_count += 1


data = {
    "prompts": prompts[:100],
    "responses": responses[:100]
}


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

# Limit the input sequence length to 256 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 1024
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

gemma_lm.fit(data, epochs=1, batch_size=4)

gemma_lm.backbone.save_lora_weights("output/model.lora.h5")
