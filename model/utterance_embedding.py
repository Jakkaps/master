import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel


class UtteranceEmbedding(nn.Module):
    """Embeds dialog utterances into a fixed-size vector."""

    def __init__(self, embed_size):
        super(UtteranceEmbedding, self).__init__()

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
        )
        model = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        self.model = get_peft_model(model, peft_config)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, x):
        attn_mask = x.ne(0).int()
        out = self.model(x, attention_mask=attn_mask)
        embeddings = out.last_hidden_state[
            :, 0, :
        ]  # Index 0 for the [CLS] token in each sequence
        return self.bn(embeddings)
