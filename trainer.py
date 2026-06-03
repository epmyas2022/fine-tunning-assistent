import modal

app = modal.App("qwen-finetune")


filename = "isaac_dataset_pirata_v2.json"

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
    gpu="A10G",
    timeout=60 * 60 * 24, # 24 horas
    volumes={"/trained-models": volume},
)
def training():
    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_sequence_length = 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct",
        max_seq_length=max_sequence_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16, # Representa cuánta información nueva o especializada puede almacenar el modelo.
        lora_alpha=16, #define cuánto peso o influencia tiene el LoRA sobre el modelo original al generar un resultado
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


    def formatting_func(example):

        return f"""### Instruction:
    {example["instruction"]}

    ### Input:
    {example["input"]}

    ### Response:
    {example["output"]}
    {tokenizer.eos_token}
"""

    data = data.map(lambda x: {"text": formatting_func(x)})

    print(data[:5])

    print("Starting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            max_steps=-1,
            warmup_steps=20,
            num_train_epochs=4,
            learning_rate=2e-4,
            logging_steps=10,
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
    training.spawn()
