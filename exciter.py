import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class ExciterFunction(Function):
    """Custom autograd function for the Exciter layer"""
    
    @staticmethod
    def forward(ctx, input, exciter_layer):
        # Store the exciter layer reference for backward pass
        ctx.exciter_layer = exciter_layer
        
        # Forward pass: clamp to [-10, 10] range (1:1 activation)
        output = torch.clamp(input, min=-10.0, max=10.0)
        
        # Only output from the N'th neuron (last neuron)
        if output.dim() > 1:
            # For batch processing, take the last neuron for each sample
            result = output[:, -1:] if output.size(1) > 1 else output
        else:
            result = output[-1:] if output.size(0) > 1 else output
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Call the exciter's custom backward method
        exciter_layer = ctx.exciter_layer
        exciter_layer.custom_backward(grad_output)
        
        # Return gradient for input (standard gradient for clamped values)
        return grad_output, None

class ExciterLayer(nn.Module):
    """Custom Exciter Layer with N neurons"""
    
    def __init__(self, n_neurons, callback_fn=None):
        super(ExciterLayer, self).__init__()
        self.n_neurons = n_neurons
        self.callback_fn = callback_fn
        
        # Initialize weights to 1.0 and bias to 0.0 as specified
        self.weight = nn.Parameter(torch.ones(n_neurons))
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        
        # Store reference to the network for callback
        self.network = None
        
    def forward(self, x):
        # Apply weights (all 1.0) and bias (all 0.0)
        # Since weights are 1 and bias is 0, this is essentially a pass-through
        output = x * self.weight + self.bias
        
        # Apply custom function with backward hook
        return ExciterFunction.apply(output, self)
    
    def custom_backward(self, grad_output):
        """Custom backward pass - called during backpropagation"""
        print(f"Exciter Layer: Custom backward called with grad_output shape: {grad_output.shape}")
        
        # Call the user-defined callback if provided
        if self.callback_fn:
            self.callback_fn(self, grad_output)
        
        # Example: Adjust weights and biases based on gradient
        # This is where you can implement custom learning logic
        with torch.no_grad():
            # Example adjustment (you can customize this)
            learning_rate = 0.01
            self.weight -= learning_rate * grad_output.mean()
            self.bias -= learning_rate * grad_output.mean() * 0.1
    
    def get_test_yhat(self, x):
        """Get test prediction with no_grad as specified"""
        if self.network is None:
            raise ValueError("Network reference not set. Call set_network() first.")
        
        with torch.no_grad():
            # Forward pass from exciter + 1 onwards
            output = x
            
            # Find exciter layer position in network
            exciter_idx = None
            for i, module in enumerate(self.network.modules()):
                if module is self:
                    exciter_idx = i
                    break
            
            if exciter_idx is None:
                raise ValueError("Exciter layer not found in network")
            
            # Continue forward pass from exciter + 1
            modules_list = list(self.network.modules())[1:]  # Skip the network container
            
            for i, module in enumerate(modules_list):
                if module is self:
                    # Apply remaining layers
                    for remaining_module in modules_list[i+1:]:
                        output = remaining_module(output)
                    break
            
            return output
    
    def set_network(self, network):
        """Set reference to the parent network for callback"""
        self.network = network

# Example callback function
def exciter_callback(exciter_layer, grad_output):
    """
    Custom callback function for exciter layer
    
    Args:
        exciter_layer: The ExciterLayer instance
        grad_output: Gradient output from backpropagation
    """
    print(f"Callback called! Grad output: {grad_output}")
    
    # Example: Get test prediction with no_grad
    try:
        # You would need to provide test input here
        # test_input = torch.randn(1, exciter_layer.n_neurons)
        # test_yhat = exciter_layer.get_test_yhat(test_input)
        # print(f"Test y_hat: {test_yhat}")
        pass
    except Exception as e:
        print(f"Test y_hat calculation failed: {e}")
    
    # Custom weight/bias adjustment logic
    with torch.no_grad():
        print(f"Current weights: {exciter_layer.weight}")
        print(f"Current biases: {exciter_layer.bias}")
        
        # Example: Custom adjustment based on gradient
        adjustment_factor = 0.005
        exciter_layer.weight -= grad_output.mean() * adjustment_factor
        exciter_layer.bias -= grad_output.mean() * adjustment_factor * 0.5

# Simple Neural Network with Exciter Layer
class SimpleNetworkWithExciter(nn.Module):
    def __init__(self, input_size, exciter_neurons, hidden_size, output_size):
        super(SimpleNetworkWithExciter, self).__init__()
        
        self.input_layer = nn.Linear(input_size, exciter_neurons)
        self.exciter = ExciterLayer(exciter_neurons, callback_fn=exciter_callback)
        self.hidden_layer = nn.Linear(1, hidden_size)  # 1 because exciter outputs 1 value
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Set network reference for exciter layer
        self.exciter.set_network(self)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.exciter(x)
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Test Program
def test_exciter_layer():
    """Test program for the Exciter Layer"""
    print("Testing Custom Exciter Layer")
    print("=" * 40)
    
    # Create network
    input_size = 5
    exciter_neurons = 3
    hidden_size = 10
    output_size = 1
    
    network = SimpleNetworkWithExciter(input_size, exciter_neurons, hidden_size, output_size)
    
    # Create sample data
    batch_size = 4
    x = torch.randn(batch_size, input_size)
    y_true = torch.randn(batch_size, output_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y_true.shape}")
    
    # Forward pass
    print("\nForward pass:")
    y_pred = network(x)
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction: {y_pred}")
    
    # Loss and backward pass
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f"\nLoss: {loss.item()}")
    
    print("\nBackward pass (this will trigger exciter callback):")
    loss.backward()
    
    # Test the no_grad functionality
    print("\nTesting no_grad functionality:")
    with torch.no_grad():
        test_input = torch.randn(1, input_size)
        test_output = network(test_input)
        print(f"Test output (no_grad): {test_output}")
    
    print("\nTest completed!")

# Additional stub functions for exciter callback extensions
def advanced_exciter_callback(exciter_layer, grad_output):
    """
    Advanced callback stub for more complex exciter behavior
    
    This is where you can implement sophisticated learning algorithms
    that adjust the exciter layer's parameters based on:
    - Gradient information
    - Network performance
    - Custom learning rules
    """
    print("Advanced callback - implement custom logic here")
    
    # Placeholder for advanced logic
    # Example: Implement reinforcement learning updates
    # Example: Implement meta-learning parameter adjustments
    # Example: Implement adaptive learning rate adjustments
    pass

def exciter_performance_monitor(exciter_layer, grad_output):
    """
    Callback stub for monitoring exciter layer performance
    """
    print("Performance monitoring callback")
    
    # Monitor gradient magnitudes
    grad_magnitude = torch.norm(grad_output)
    print(f"Gradient magnitude: {grad_magnitude.item()}")
    
    # Monitor weight changes
    weight_magnitude = torch.norm(exciter_layer.weight)
    print(f"Weight magnitude: {weight_magnitude.item()}")

if __name__ == "__main__":
    test_exciter_layer()
