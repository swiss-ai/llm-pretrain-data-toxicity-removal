from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.base import PipelineStep
#from datatrove.executor.slurm_nodes import SlurmPipelineNodeExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

class DetoxifyScorer(PipelineStep):
    name = "Toxicity"
    type = "ContentFilter"

    def __init__(self,
                 model: str="multilingual",
                 batch_size: int=4096,
                 use_half_precision: bool=False):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.use_half_precision = use_half_precision

    def predict(self, model, data,
                micro_batch_size=64):
        if isinstance(data, list) and len(data) > micro_batch_size:
            N = len(data)
            data = [data[i*64:min((i+1)*64, N)] for i in range(N//64)]
            score = None
            for data_batch in data:
                s = model.predict(data_batch)
                if score is None:
                    score = s
                else:
                  for k,v in score.items():
                    score[k].extend(s[k])
        else:
            score = model.predict(data)
        if isinstance(score, dict):
            score = max(list(score.values()))
        elif isinstance(score, list):
            score = [max(list(s.values())) for s in score]
        return score

    def run(self, data, rank: int = 0, world_size: int = 1,
            micro_batch_size: int = 64):
        import torch
        from detoxify import Detoxify
        from datatrove.utils.batching import batched

        device = f"cuda:{rank % torch.cuda.device_count()}"
        model = Detoxify(self.model, device=device)
        print(f"Rank {rank}, device: {model.device}")

        for batch in batched(data, self.batch_size):
            texts = [d.text for d in batch]
            toxic_scores = [self.predict(model, t.split("\n"), micro_batch_size) for t in texts]

            for document, scores in zip(batch, toxic_scores):
                document.metadata["toxic_scores"] = scores
                yield document

LANGS = {"french": "fr",
         "italian": "it",
         "russian": "ru",
         "portuguese": "pt",
         "spanish": "es",
         "turkish": "tr"}

# Add cli params
import argparse
args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--data_dir', default='s3://fineweb-multilingual-v1/experiments/filtering/final-tail', type=str)
args_parser.add_argument('--output_dir', default='/iopsstor/scratch/cscs/fsimin/mfineweb_toxic_filtering/detoxify/data', type=str)
args_parser.add_argument('--language', default='fr', type=str)

if __name__ == "__main__":
    args = args_parser.parse_args()
    language = args.language
    language_code = LANGS[language]
    pipeline = [
        JsonlReader(
            f"s3://fineweb-multilingual-v1/experiments/filtering/final-tail/{language}/",
            text_key="text",
            limit=25_000,
            shuffle_files=True,
        ),
        DetoxifyScorer(
            model="multilingual",
            batch_size=4096,
        ),
        JsonlWriter(f"{args.output_dir}/{language_code}"),
    ]

    LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"/iopsstor/scratch/cscs/fsimin/mfineweb_toxic_filtering/detoxify/logs/{language_code}",
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