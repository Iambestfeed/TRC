import keras
import keras_hub

gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("my_model")

gemma_lm.summary()

gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.backbone.load_lora_weights('output/model.lora.h5')

gemma_lm.summary()
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
