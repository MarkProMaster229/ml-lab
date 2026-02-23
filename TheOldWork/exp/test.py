import sys
sys.path.append("/home/chelovek/Документы/work/ml-lab/exp/tensor")

import tensor_module  
import torch


vec = [0.1] * (30*40)

t = tensor_module.Tensor(vec, 30, 40)

emb = t.embeddingLookup(5)
print("Embedding токена 5:", emb)


batch = t.Batch([0, 5, 12])
print("Batch токенов 0,5,12:", batch)


all_tokens = list(range(30))
batches = t.createBatches(all_tokens, 10)
print("Количество батчей:", len(batches))
print("Первый батч:", batches[0])
tensor_batch = torch.tensor(batch, dtype=torch.float32).view(3, 40)

linear = torch.nn.Linear(40, 10)


output = linear(tensor_batch)
print("Output Linear:", output)

output.sum().backward()