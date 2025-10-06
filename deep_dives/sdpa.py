import torch
import torch.nn.functional as F

"""
Simple example to illustrate the attention weight calculation in scaled dot-product attention (SDPA).
"""

# each row is a token embedding, and each column is a dimension of the embedding
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print("Q @ K.T -- How similar a query (row) matches each key (column)")

# K.T means each column is a token embedding, so dot product is some
# similarity score in direction of embedding in embedding space
scores = Q @ K.T
print(scores)  # query 1 matches key 1, but is orthogonal to key 2, and vice versa
scores /= torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
print("\nScaled scores (divided by sqrt(d_k)):")
print(scores)

# each row sums to 1, weight for each key (how much to attend to each key)
attention = F.softmax(scores, dim=-1)
print("\nAttention weights (softmax over scaled scores):")
print(attention)

# as a tangent, let's see how masking affects the attention weights
mask = torch.tensor([[1, 0], [1, 1]])  # mask out second token score
masked_scores = scores.masked_fill(mask == 0, -float("inf"))
masked_attention = F.softmax(masked_scores, dim=-1)
print("\nMasked attention weights (masking out some keys):")
print(masked_attention)  # query 1 can't attend to key 2 at all since masked

out = attention @ V  # each value gets weighted by how much attention to pay to it
print("\nOriginal values:")
print(V)
print("Output (weighted sum of values):")
print(out)
# Notice how value[0] becomes a bit closer to value[1] because query[0] attends a bit to key[1] too
# and vice versa. but still mostly attends to its own value
