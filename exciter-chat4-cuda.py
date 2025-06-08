import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random

# Automatically choose CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExciterFunction(Function):
    """Custom autograd function for the Exciter layer"""

    @staticmethod
    def forward(ctx, input, exciter_layer):
        ctx.exciter_layer = exciter_layer
        ctx.save_for_backward(input)
        output = torch.clamp(input, min=-10.0, max=10.0)
        return output  # Return all neurons, no slicing

    @staticmethod
    def backward(ctx, grad_output):
        exciter_layer = ctx.exciter_layer
        input, = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[input < -10.0] = 0
        grad_input[input > 10.0] = 0

        exciter_layer.custom_backward(grad_output)
        return grad_input, None

class ExciterLayer(nn.Module):
    def __init__(self, n_neurons, callback_fn=None):
        super(ExciterLayer, self).__init__()
        self.n_neurons = n_neurons
        self.callback_fn = callback_fn
        self.weight = nn.Parameter(torch.ones(n_neurons))
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.network = None

    def forward(self, x):
        output = x * self.weight + self.bias
        return ExciterFunction.apply(output, self)

    @torch.no_grad()
    def custom_backward(self, grad_output):
        print(f"Exciter Layer: Custom backward called with grad_output shape: {grad_output.shape}")

        # --- CALLBACK LOGIC ---
        if self.callback_fn:
            self.callback_fn(self, grad_output)

        # Simple example weight update
        grad = grad_output.mean().item()
        self.weight -= 0.01 * grad
        self.bias -= 0.001 * grad

    def get_test_yhat(self, x):
        if self.network is None:
            raise ValueError("Network reference not set. Call set_network() first.")

        with torch.no_grad():
            output = x
            modules_list = list(self.network.children())

            found = False
            for i, module in enumerate(modules_list):
                if module is self:
                    found = True
                    for remaining_module in modules_list[i+1:]:
                        output = remaining_module(output)
                    break

            if not found:
                raise ValueError("Exciter layer not found in network")

            return output

    def set_network(self, network):
        self.network = network

def exciter_callback(exciter_layer, grad_output):
    print(f"Callback called! Grad output: {grad_output}")

    # --- Neuron-specific callback ---
    neuron_index_to_trigger = 1  # Can also be randomized
    if exciter_layer.n_neurons > neuron_index_to_trigger:
        print(f"Neuron E{neuron_index_to_trigger} triggering y_hat computation...")

        # Create a synthetic test input to feed into the network
        try:
            test_input_size = exciter_layer.network.input_layer.in_features
            test_input = torch.randn(1, test_input_size).to(exciter_layer.weight.device)

            # Call get_test_yhat to simulate Exciter neuron probing the output
            test_yhat = exciter_layer.get_test_yhat(test_input)
            print(f"Test y_hat from neuron E{neuron_index_to_trigger}: {test_yhat.cpu()}")  # move to CPU for safe printing
        except Exception as e:
            print(f"Failed to compute test_yhat: {e}")

    # --- Regular adjustment logic ---
    with torch.no_grad():
        print(f"Current weights: {exciter_layer.weight}")
        print(f"Current biases: {exciter_layer.bias}")

        grad = grad_output.mean().item()
        exciter_layer.weight -= 0.005 * grad
        exciter_layer.bias -= 0.0025 * grad

class MaskedLinear(nn.Linear):
    def __init__(self, size):
        super().__init__(size, size)
        self.register_buffer('mask', torch.eye(size))  # Only diagonal weights active

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

class SimpleNetworkWithExciter(nn.Module):
    def __init__(self, input_size, exciter_neurons, hidden_size, output_size):
        super(SimpleNetworkWithExciter, self).__init__()
        self.input_layer = nn.Linear(input_size, exciter_neurons)
        self.exciter = ExciterLayer(exciter_neurons, callback_fn=exciter_callback)
        self.use_passthrough = True  # Toggle for passthrough vs masked linear
        self.hidden_layer = nn.Identity() if self.use_passthrough else MaskedLinear(exciter_neurons)
        self.output_layer = nn.Linear(exciter_neurons, output_size)
        self.exciter.set_network(self)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.exciter(x)
        x = F.relu(self.hidden_layer(x))  # Pure activation after Exciter
        x = self.output_layer(x)
        return x

def test_exciter_layer():
    print("Testing Custom Exciter Layer")
    print("=" * 40)

    input_size = 5
    exciter_neurons = 3
    hidden_size = 3  # Must match exciter_neurons for 1:1 masking
    output_size = 1

    network = SimpleNetworkWithExciter(input_size, exciter_neurons, hidden_size, output_size).to(device)

    batch_size = 4
    x = torch.randn(batch_size, input_size).to(device)
    y_true = torch.randn(batch_size, output_size).to(device)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y_true.shape}")

    print("\nForward pass:")
    y_pred = network(x)
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction: {y_pred}")

    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f"\nLoss: {loss.item()}")

    print("\nBackward pass (this will trigger exciter callback):")
    loss.backward()

    print("\nTesting no_grad functionality:")
    with torch.no_grad():
        test_input = torch.randn(1, input_size).to(device)
        test_output = network(test_input)
        print(f"Test output (no_grad): {test_output}")

    print("\nTest completed!")

if __name__ == "__main__":
    test_exciter_layer()

