import os
import pandas
import json

# --- Environment Setup for JAX/TPU ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras
import keras_hub # Using keras_hub as per your current script

# --- Configuration ---
MODEL_PRESET = "my_model" # Your custom Gemma model preset
LORA_RANK = 32
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 4 # Still potentially large for TPU with this sequence length and long prompts.
               # If OOM occurs after fixing KeyError, reduce this to 2 or 1 first.
EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DATA_FILE_PATH = "data/PhoMT_training.csv"
SAMPLE_SIZE = None
# SAMPLE_SIZE = 200 # For a very quick test run

# --- Prompt Definition ---
SOURCE_LANGUAGE = "English"
TARGET_LANGUAGE = "Vietnamese"
CUSTOM_DICTIONARY_STR = "None provided for this task."
DESIRED_STYLE = (
    "Maintain a poetic and elegant tone, enhancing the translation "
    "with stylistic elements that resonate with the original’s intellectual depth."
)

PROMPT_TEMPLATE_PREFIX = f"""Translate this from {SOURCE_LANGUAGE} into {TARGET_LANGUAGE}, ensuring to use the provided custom dictionary for specified terms and maintaining the poetic style as described.
Custom Dictionary: {CUSTOM_DICTIONARY_STR}
Desired Style: {DESIRED_STYLE}
{SOURCE_LANGUAGE}: """

PROMPT_TEMPLATE_SUFFIX = f"\n{TARGET_LANGUAGE}:"

def main():
    print(f"Using Keras backend: {keras.backend.backend()}")
    print(f"Keras version: {keras.__version__}")
    if hasattr(keras_hub, '__version__'): # Check if __version__ attribute exists
        print(f"KerasHub version: {keras_hub.__version__}")
    else:
        print("KerasHub version attribute not found (this is okay for some installations).")


    # --- Load and Prepare Data ---
    print(f"Loading data from {DATA_FILE_PATH}...")
    try:
        if not os.path.exists(DATA_FILE_PATH) and DATA_FILE_PATH == "data/PhoMT_training.csv":
             # Try relative path if default path doesn't exist (e.g., running from parent dir)
            alt_path = "PhoMT_training.csv"
            if os.path.exists(alt_path):
                print(f"Default path {DATA_FILE_PATH} not found, using alternative path {alt_path}")
                data_file_to_load = alt_path
            else:
                raise FileNotFoundError(f"Neither {DATA_FILE_PATH} nor {alt_path} found.")
        else:
            data_file_to_load = DATA_FILE_PATH

        if SAMPLE_SIZE:
            df = pandas.read_csv(data_file_to_load, nrows=SAMPLE_SIZE)
        else:
            df = pandas.read_csv(data_file_to_load)
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found. {e}")
        print("Please ensure 'PhoMT_training.csv' is in the specified path or project root.")
        print("Creating a dummy PhoMT_training.csv for testing purposes...")
        dummy_data = {
            'en': [
                "Hello, world!", "This is a test.", "Another sentence for translation.",
                "The quick brown fox jumps over the lazy dog.", "Programming is fun."
            ],
            'vi': [
                "Chào thế giới!", "Đây là một bài kiểm tra.", "Một câu khác để dịch.",
                "Con cáo nâu nhanh nhẹn nhảy qua con chó lười biếng.", "Lập trình rất vui."
            ]
        }
        df = pandas.DataFrame(dummy_data)
        if SAMPLE_SIZE and SAMPLE_SIZE <= len(dummy_data['en']):
            df = df.head(SAMPLE_SIZE)
        elif SAMPLE_SIZE:
             print(f"Warning: Dummy data has fewer than {SAMPLE_SIZE} rows. Using all dummy data.")
        print("Dummy data created and loaded.")


    print(f"Loaded {len(df)} samples.")

    formatted_prompts_input = [] # This will be the "prompt" part for the model
    formatted_responses_target = [] # This will be the "response" part the model learns to generate

    for index, row in df.iterrows():
        en_sentence = str(row['en'])
        vi_sentence = str(row['vi'])

        # The input to the model during training is the full string including the start of the target.
        # The preprocessor will handle masking the loss on the prompt part.
        # `keras_hub.Gemma3CausalLM` expects the full concatenated string as part of its "prompts" key input
        # and the "responses" key should contain what it needs to predict.
        # Typically for Causal LM fine-tuning with prompt/response pairs:
        # prompt = "Instruction: Do X. Input: Y."
        # response = "Output: Z"
        # The model is fed "Instruction: Do X. Input: Y. Output: Z"
        # and learns to predict "Output: Z" given "Instruction: Do X. Input: Y."
        # The `keras_hub.Gemma3CausalLMPreprocessor` expects 'prompts' and 'responses' keys.
        # 'prompts' should be the part *before* the model starts generating.
        # 'responses' should be the part the model *should* generate.

        # Correct way for keras_hub.Gemma3CausalLM:
        # "prompts" key takes the text leading up to the desired generation.
        # "responses" key takes the text that should be generated.
        current_prompt_text = f"{PROMPT_TEMPLATE_PREFIX}{en_sentence}{PROMPT_TEMPLATE_SUFFIX}"
        current_response_text = vi_sentence

        formatted_prompts_input.append(current_prompt_text)
        formatted_responses_target.append(current_response_text)

    # *** THIS IS THE CRITICAL FIX for the KeyError ***
    data_for_ft = {
        "prompts": formatted_prompts_input,    # Use "prompts" (plural)
        "responses": formatted_responses_target # "responses" (plural) is correct
    }

    if not data_for_ft["prompts"] or not data_for_ft["responses"]:
        print("ERROR: No data loaded into formatted_prompts_input or formatted_responses_target. Exiting.")
        return

    print(f"First formatted 'prompts' example for model:\n{data_for_ft['prompts'][0]}")
    print(f"First formatted 'responses' example for model:\n{data_for_ft['responses'][0]}")


    # --- Load Model ---
    print(f"Loading model: {MODEL_PRESET}")
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(
        MODEL_PRESET,
        # Load the preprocessor directly if separate, or it's part of the model
    )
    # The preprocessor is typically attached to the model instance in keras_hub
    if gemma_lm.preprocessor is None:
        print("Warning: Model does not have a preprocessor attached. This might cause issues.")
        # Attempt to load it explicitly if that's how 'my_model' is structured
        # This part is speculative, as 'my_model' structure is unknown
        # from keras_hub.models.gemma3 import Gemma3CausalLMPreprocessor
        # gemma_lm.preprocessor = Gemma3CausalLMPreprocessor.from_preset(MODEL_PRESET) # This might not be right for 'my_model'
    else:
        print(f"Preprocessor found: {type(gemma_lm.preprocessor)}")

    gemma_lm.preprocessor.sequence_length = SEQUENCE_LENGTH # Set sequence length on the model's preprocessor
    gemma_lm.summary()

    # --- Enable LoRA ---
    print(f"Enabling LoRA with rank={LORA_RANK}")
    gemma_lm.backbone.enable_lora(rank=LORA_RANK)
    gemma_lm.summary()


    # --- Compile Model ---
    print("Compiling model...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale", "token_embedding/embeddings"])

    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # --- Fit Model ---
    print(f"Starting training with batch_size={BATCH_SIZE}, epochs={EPOCHS}...")
    gemma_lm.fit(
        data_for_ft,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # --- Save LoRA Weights ---
    lora_output_dir = "output_lora_phomt"
    os.makedirs(lora_output_dir, exist_ok=True)
    # Use a more descriptive name for LoRA weights if saving only LoRA
    lora_weights_path = os.path.join(lora_output_dir, f"{MODEL_PRESET.replace('/','_')}_lora_rank{LORA_RANK}_seq{SEQUENCE_LENGTH}_bs{BATCH_SIZE}.lora.h5")
    print(f"Saving LoRA weights to {lora_weights_path}")
    gemma_lm.backbone.save_lora_weights(lora_weights_path) # Correct for saving only LoRA with keras_hub Gemma

    print("Training finished and LoRA weights saved.")

    # --- Example Inference (Optional) ---
    print("\n--- Example Inference ---")
    test_english_sentence = "The sun shines brightly in the summer."
    # For inference, the prompt should NOT include the target language suffix if the model is trained to generate it.
    # The `Gemma3CausalLM` from `keras_hub` usually expects the prompt for `generate` to be just the input.
    # It will append its generation template if needed.
    # However, since our training data `prompts` key includes the `PROMPT_TEMPLATE_SUFFIX`,
    # the model is learning to generate *after* that suffix.
    # So, for consistency with training, the inference prompt should also include it.
    inference_prompt_for_generate = f"{PROMPT_TEMPLATE_PREFIX}{test_english_sentence}{PROMPT_TEMPLATE_SUFFIX}"

    print(f"Inference Prompt for generate():\n{inference_prompt_for_generate}")

    # Access tokenizer from the model's preprocessor
    if gemma_lm.preprocessor and hasattr(gemma_lm.preprocessor, 'tokenizer'):
        # *** FIX for inference tokenizer access ***
        num_prompt_tokens = len(gemma_lm.preprocessor.tokenizer.tokenize(inference_prompt_for_generate))
        max_generate_length = num_prompt_tokens + 64 # Generate up to 64 new tokens
        print(f"Number of prompt tokens: {num_prompt_tokens}, Max generate length: {max_generate_length}")
    else:
        print("Warning: Could not access tokenizer for inference length calculation. Using default.")
        max_generate_length = SEQUENCE_LENGTH # Fallback, might be too long or too short

    try:
        output_text = gemma_lm.generate(inference_prompt_for_generate, max_length=max_generate_length)
        print(f"\nGenerated Vietnamese (raw model output):\n{output_text}")

        # Post-processing:
        # The model's output will likely contain the input prompt. We need to extract the generated part.
        # Since `inference_prompt_for_generate` is what we fed, the new text starts after it.
        if output_text.startswith(inference_prompt_for_generate):
            generated_translation = output_text[len(inference_prompt_for_generate):].strip()
        else:
            # Fallback if the model output is unexpected (e.g., doesn't include the full prompt)
            # This might happen if the generate method has its own template handling.
            # A more robust way would be to see if PROMPT_TEMPLATE_SUFFIX is in output_text
            # and split after the *last* occurrence if the model repeats it.
            if PROMPT_TEMPLATE_SUFFIX in output_text:
                # Find the last occurrence of the suffix if the prompt itself is part of the output
                parts = output_text.split(PROMPT_TEMPLATE_SUFFIX)
                if len(parts) > 1 :
                    generated_translation = parts[-1].strip() # Take the last part after the final suffix
                else: # Suffix not found or only at the beginning (unlikely for generation)
                    generated_translation = output_text # Could be that the model only outputted the translation
            else:
                generated_translation = output_text # Assume raw output is just the translation

        print(f"\nGenerated Vietnamese (extracted):\n{generated_translation}")

    except Exception as e:
        print(f"Error during example inference: {e}")
        import traceback
        traceback.print_exc()
        print("Inference might require specific JAX configurations or model state.")

if __name__ == "__main__":
    main()
