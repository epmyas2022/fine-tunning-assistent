import modal

app = modal.App("qwen-finetune")


filename = "alpaca_data_cleaned_spanish.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(
        f"data/{filename}",
        f"/root/data/{filename}",
        copy=True,
    )
    .pip_install("unsloth", "transformers", "datasets", "trl", "accelerate", "peft")
)

volume = modal.Volume.from_name("trained-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    volumes={"/trained-models": volume},
)
def training():
    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_sequence_length = 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=max_sequence_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,  # Taza de desajuste
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    data = load_dataset("json", data_files=f"data/{filename}", split="train")


    def transform_data(data):
        alpaca_data = []
        messages = {row["message_id"]: row for row in data}
        for row in data:
            if row["role"] != "assistant":
                continue

            parent_id = row["parent_id"]

            if parent_id is None:
                continue

            parent = messages.get(parent_id)

            if not parent:
                continue

            if parent["role"] != "prompter":
                continue

            alpaca_data.append({"instruction": parent["text"], "input": "", "output": row["text"]})
        return alpaca_data

    def formatting_func(example):

        return f"""### Instruction:
    {example["instruction"]}

    ### Input:
    {example["input"]}

    ### Response:
    {example["output"]}
    {tokenizer.eos_token}
"""

    #data = Dataset.from_list(transform_data(data))
    data = data.map(lambda x: {"text": formatting_func(x)})

    print(data[:5])

    print("Starting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,
            learning_rate=5e-5,
            logging_steps=1,
            output_dir="./tmp/outputs",
        ),
    )

    trainer.train()

    # save adapters

    model.save_pretrained("/trained-models/qwen-lora")
    tokenizer.save_pretrained("/trained-models/qwen-lora")

    ## fusionar

    model.save_pretrained_merged(
        "/trained-models/qwen-lora-merged", tokenizer, save_method="merged_16bit"
    )

    volume.commit()

    print("Training completed and model saved to volume.")


@app.local_entrypoint()
def main():
    training.remote()
