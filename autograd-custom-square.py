import torch

class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # store input for backward pass
        return input * input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        return grad_input

# Use it
x = torch.tensor([2.0], requires_grad=True)
y = MySquare.apply(x)
y.backward()

print(x.grad)  # prints: tensor([4.])

