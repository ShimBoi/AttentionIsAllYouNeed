import torch

"""
This file focuses on providing examples to help understand why embedding scaling is necessary
and how it affects the training dynamics of transformer models.

- scaling by sqrt(d_k) helps to keep the variance of the dot product stable even as the 
    dimension of the embeddings increase.
"""

# generate random data from gaussian with mean 0 and std 1
data_1 = torch.randn(256, 256)
data_2 = torch.randn(512, 512)
data_3 = torch.randn(1024, 1024)

# As the dimension increases, the variance of the dot product increases
dot_product_1 = torch.matmul(data_1, data_1.T)
dot_product_2 = torch.matmul(data_2, data_2.T)
dot_product_3 = torch.matmul(data_3, data_3.T)

print("WITHOUT SCALING\n#################################")
print("Dot product variance for 256-dim data:", dot_product_1.var().item())
print("Dot product variance for 512-dim data:", dot_product_2.var().item())
print("Dot product variance for 1024-dim data:", dot_product_3.var().item())

# Now scale the dot product by sqrt(d_k)
scaled_dot_product_1 = dot_product_1 / torch.sqrt(torch.tensor(256.0))
scaled_dot_product_2 = dot_product_2 / torch.sqrt(torch.tensor(512.0))
scaled_dot_product_3 = dot_product_3 / torch.sqrt(torch.tensor(1024.0))

print("\nWITH SCALING\n#################################")
print(
    "Scaled dot product variance for 256-dim data:", scaled_dot_product_1.var().item()
)
print(
    "Scaled dot product variance for 512-dim data:", scaled_dot_product_2.var().item()
)
print(
    "Scaled dot product variance for 1024-dim data:", scaled_dot_product_3.var().item()
)

# the variances matter since they affect output of softmax
data = torch.tensor([-1.0, 0.0, 1.0])
print("\nSOFTMAX OUTPUT\n#################################")
print("Softmax output for data with variance 1:", torch.softmax(data, dim=0))
print(
    "Softmax output for data with variance 1000:", torch.softmax(data * 1000, dim=0)
)  # basically becomes a one-hot vector!
