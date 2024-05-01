import torch
from torch import Tensor
import torch.nn.functional as F

def create_nodes(chats: List):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    max_sen_len = 0
    max_nodes = 0
    tokenized = []
    for c1, c2 in shuffled_chats:
        t1, t2 = model.tokenize(c1)['input_ids'], model.tokenize(c2)['input_ids']
        max_sen_len = max(max_sen_len, t1.size(1), t2.size(1))
        max_nodes = max(max_nodes, t1.size(0), t2.size(0))
        tokenized.append([
            t1, t2
        ])

    node_tensor = torch.zeros(len(tokenized), 2, max_nodes, max_sen_len)
    for i, (c1, c2) in enumerate(tokenized): 
        p1 = F.pad(c1, (0, max_sen_len - c1.size(1), 0, max_nodes - c1.size(0))) 
        p2 = F.pad(c2, (0, max_sen_len - c2.size(1), 0, max_nodes - c2.size(0)))

        node_tensor[i] = torch.stack([p1, p2])