import torch

"""
This file showcases why a reidual block is neccessary and how the Jacobian < 1 causes vanishing gradients
that can be solved by residual connections.
"""

x = torch.randn(3, 1, requires_grad=True)
W = torch.randn(3, 3) * 0.1

# calculate the Jacobian (what happens under the hood with autograd during backprop)
J = torch.autograd.functional.jacobian(lambda x: torch.relu(W @ x), x)
# remove extra dimensions since autograd jacobian calculates for batches
J = J.squeeze(-1).squeeze(1)

print("Jacobian < 1 causes vanishing gradients")
print("Jacobian:\n", J)
print("Spectral norm:", torch.linalg.norm(J, 2).item())  # largest singular value < 1

y = torch.relu(W @ x)
y.sum().backward()
print("Gradient without residual:\n", x.grad)  # very small gradients


x.grad = None
# add residual connection
J = torch.autograd.functional.jacobian(lambda x: x + torch.relu(W @ x), x)
J = J.squeeze(-1).squeeze(1)

print("\n\n\nJacobian > 1 helps preserve gradients")
print("Jacobian with residual:\n", J)
print("Spectral norm:", torch.linalg.norm(J, 2).item())  # spectral norm > 1 now

y = x + torch.relu(W @ x)
y.sum().backward()
print("Gradient with residual:\n", x.grad)  # better gradient signal
