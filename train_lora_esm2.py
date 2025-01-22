import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pandas as pd
from datetime import datetime
from sklearn import metrics
from torch.utils.data import Dataset
from transformers import (
    EsmForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator


def train_protein_model():
    # Initialize accelerator
    accelerator = Accelerator()
    train_dataframe = pd.read_csv("./Data/eSol_train.csv", sep=",")
    test_dataframe = pd.read_csv("./Data/eSol_test.csv", sep=",")
    tokenizer = AutoTokenizer.from_pretrained(
        "./facebook/esm2_t6_8M_UR50D"
    )

    class ProteinDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=512):
            self.sequences = dataframe["sequence"].tolist()
            self.solubilities = dataframe["solubility"].tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx][: self.max_length]
            solubility = self.solubilities[idx]
            encoding = self.tokenizer(
                sequence,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            encoding["labels"] = solubility
            return encoding

    train_dataset = ProteinDataset(train_dataframe, tokenizer)
    test_dataset = ProteinDataset(test_dataframe, tokenizer)
    train_dataset, test_dataset = accelerator.prepare(train_dataset, test_dataset)

    def model_init(trial):
        base_model = EsmForSequenceClassification.from_pretrained(
            "./facebook/esm2_t6_8M_UR50D", num_labels=1
        )
        config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="all",
        )
        lora_model = get_peft_model(base_model, config)
        return accelerator.prepare(lora_model)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        r2 = metrics.r2_score(labels, logits)
        return {"r2": r2}

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"ESOL-esm2_t6_8M-finetuned-lora_{timestamp_str}"

    args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        save_strategy="epoch",
        label_names=["labels"],
        report_to=[],
    )

    trainer = Trainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    model = model_init(None)
    trainer.train()

    # Explicitly save the model's configuration
    model.config.save_pretrained(output_dir)

    # Save the model
    trainer.save_model(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    train_protein_model()
