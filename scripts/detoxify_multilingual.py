from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
#from datatrove.executor.slurm_nodes import SlurmPipelineNodeExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import torch
import math

class RobertaClassifier:
    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)

    def chunk_text(self, text, chunk_size=512):
        inputs = self.tokenizer(text, return_tensors="pt")
        N = inputs.input_ids.shape[1]
        if N > chunk_size:
            input_ids_chunks = []
            attention_mask_chunks = []
            for i in range(0, math.ceil(inputs.input_ids.shape[1]/chunk_size)):
                chunk = inputs.input_ids[:, i:min(i+chunk_size, N)]
                mask = inputs.attention_mask[:, i:min(i+chunk_size, N)]
                if chunk.shape[1] < chunk_size:
                    padded_tokens = self.tokenizer.pad(inputs, max_length=chunk_size, padding="max_length",return_tensors="pt")
                    input_ids_chunks.append(padded_tokens.input_ids)
                    attention_mask_chunks.append(padded_tokens.attention_mask)
                else:
                    input_ids_chunks.append(chunk)
                    attention_mask_chunks.append(mask)
            return torch.cat(input_ids_chunks, dim=0), torch.cat(attention_mask_chunks, dim=0)
        else:
            padded_tokens = self.tokenizer.pad(inputs, max_length=chunk_size, padding="max_length",return_tensors="pt")
            return padded_tokens.input_ids, padded_tokens.attention_mask

    def predict(self, text, chunk_size=512, output_hidden_states=True, device="cuda:0"):
        self.model.eval()
        self.model.to(device)
        input_ids = []
        attention_mask = []
        input_ids, attention_mask = self.chunk_text(text, chunk_size)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
            logits = outputs.logits
        if output_hidden_states:
            return max(outputs.logits.softmax(dim=1)[:, 0].cpu().numpy()), outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        else:
            return max(outputs.logits.softmax(dim=1)[:, 0].cpu().numpy()), None

class DetoxifyScorer(PipelineStep):
    name = "Toxicity"
    type = "ContentFilter"

    def __init__(self,
                 model,
                 tokenizer,
                 batch_size: int=4096,
                 use_half_precision: bool=False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.detoxifier = RobertaClassifier(self.model, self.tokenizer)
        self.use_half_precision = use_half_precision

    def predict(self, data,
                detoxifier,
                chunk_size: int=512,
                micro_batch_size: int=64,
                output_hidden_states: bool=True,
                device="cuda:0"):
        if isinstance(data, list) and len(data) > micro_batch_size:
            N = len(data)
            data = [data[i*micro_batch_size:min((i+1)*micro_batch_size, N)] for i in range(N//micro_batch_size)]
            scores = []
            hiddens = []
            for data_batch in data:
                score, hidden = detoxifier.predict(data_batch, chunk_size=chunk_size, output_hidden_states=output_hidden_states, device=device)
                scores.append(score)
                if output_hidden_states:
                    hiddens.append(hidden)
        else:
            scores, hiddens = detoxifier.predict(data, chunk_size=chunk_size, output_hidden_states=output_hidden_states, device=device)
        return scores, hiddens

    def run(self, data, rank: int = 0, world_size: int = 1,
            micro_batch_size: int = 64,
            output_hidden_states: bool = True):
        import torch
        from datatrove.utils.batching import batched
        device = f"cuda:{rank % torch.cuda.device_count()}"
        print(f"Rank {rank}, device: {device}")

        for batch in batched(data, self.batch_size):
            texts = [d.text for d in batch]
            toxic_scores = []
            hiddens = []
            for t in texts:
                s, h = self.predict(self.detoxifier, t, chunk_size=512, output_hidden_states=output_hidden_states, device=device)
                toxic_scores.append(s)
                if output_hidden_states:
                    hiddens.append(h)

            for document, score, hidden in zip(batch, toxic_scores, hiddens):
                document.metadata["toxic_score"] = score
                document.metadata["hidden"] = hidden
                yield document

LANGS = {"french": "fr",
         "italian": "it",
         "russian": "ru",
         "portuguese": "pt",
         "spanish": "es",
         "turkish": "tr"}

import argparse
args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--model_dir', default="/users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual", type=str)
args_parser.add_argument('--data_dir', default="s3://fineweb-multilingual-v1/experiments/filtering/final-tail", type=str)
args_parser.add_argument('--output_dir', default="/iopsstor/scratch/cscs/fsimin/mfineweb_toxic_filtering/detoxify/data", type=str)
args_parser.add_argument('--log_dir', default="/iopsstor/scratch/cscs/fsimin/mfineweb_toxic_filtering/detoxify/logs", type=str)
args_parser.add_argument('--language', default='fr', type=str)

if __name__ == "__main__":
    args = args_parser.parse_args()
    language = args.language
    language_code = LANGS[language]
    
    MULTILINGUAL_MODEL_PATH = args.model_dir
    model_multilingual = AutoModelForSequenceClassification.from_pretrained(MULTILINGUAL_MODEL_PATH)
    tokenizer_multilingual = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    # for language, language_code in LANGS:
    pipeline = [
        JsonlReader(
            f"{args.data_dir}/{language}/",
            text_key="text",
            limit=25_000,
            shuffle_files=True,
        ),
        DetoxifyScorer(
            model=model_multilingual,
            tokenizer=tokenizer_multilingual,
            batch_size=4096,
        ),
        JsonlWriter(f"{args.output_dir}/{language_code}"),
    ]

    LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{args.log_dir}/{language_code}",
        tasks=4,
        # cpus_per_node=4,
        # time="01:00:00",
        # partition="normal",
        # cpus_per_task=1,
        # sbatch_args={"reservation": "sai-a06", "account": "a06"},
        # srun_args={"reservation": "sai-a06", "account": "a06"},
        # sbatch_args={"environment": "sbert"},
        # srun_args={"environment": "sbert"},
        # environment="sbert"
    ).run()