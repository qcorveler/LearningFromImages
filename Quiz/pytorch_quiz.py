# We want to find the minimum of two simple functions.
# 1. f(x) = x^2 + 1
# 2. f(x) = 2*x^2 + 6*x
import torch

# Why do we need requires_grad here? Research this!
x = torch.tensor([2.0], requires_grad=True)

# Define the SGD optimizer with a learning rate of 0.1!
optimizer = torch.optim.SGD([x], lr=0.1)

# Optimization steps
for step in range(100):
    # Set gradients to zero
    optimizer.zero_grad()

    # Execute the function    
    y = x**2 + 1            
    y = 2*x**2 + 6*x

    # Backpropagation step to compute the gradients
    y.backward()          

    # With the gradients, let the optimizer take the next step.
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, f(x) = {y.item():.4f}")

# Final result please copy/paste into Moodle
# "Minimum at x = ..."
print(f"Minimum at x = {x.item():.4f}")