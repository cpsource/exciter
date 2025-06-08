import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer
import numpy as np


class AstrocyteMemoryModule(nn.Module):
    """
    Associative memory module that can be integrated into any PyTorch neural network.
    Works by modulating neural activations based on stored patterns.
    """
    
    def __init__(self, neural_dim=512, memory_size=1000, num_heads=8):
        super().__init__()
        
        # Memory banks
        self.register_buffer('memory_keys', torch.zeros(memory_size, neural_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, neural_dim))
        self.register_buffer('memory_used', torch.zeros(memory_size, dtype=torch.bool))
        
        self.memory_ptr = 0
        self.memory_size = memory_size
        self.neural_dim = neural_dim
        
        # Learnable parameters for memory operations
        self.key_proj = nn.Linear(neural_dim, neural_dim)
        self.value_proj = nn.Linear(neural_dim, neural_dim)
        self.query_proj = nn.Linear(neural_dim, neural_dim)
        
        # Multi-head attention for memory retrieval
        self.memory_attn = nn.MultiheadAttention(neural_dim, num_heads, batch_first=True)
        
        # Gating mechanism - learns when to use memory
        self.memory_gate = nn.Sequential(
            nn.Linear(neural_dim * 2, neural_dim),
            nn.ReLU(),
            nn.Linear(neural_dim, neural_dim),
            nn.Sigmoid()
        )
        
        # Integration layer - combines neural and memory signals
        self.integration = nn.Sequential(
            nn.Linear(neural_dim * 2, neural_dim * 2),
            nn.ReLU(),
            nn.Linear(neural_dim * 2, neural_dim)
        )
        
    def write_memory(self, key, value):
        """Write to memory (during training)."""
        with torch.no_grad():
            idx = self.memory_ptr % self.memory_size
            self.memory_keys[idx] = key.detach()
            self.memory_values[idx] = value.detach()
            self.memory_used[idx] = True
            self.memory_ptr += 1
    
    def read_memory(self, query, k=5):
        """Read from memory using attention mechanism."""
        # Get used memories
        if not self.memory_used.any():
            return None
            
        used_keys = self.memory_keys[self.memory_used]
        used_values = self.memory_values[self.memory_used]
        
        # Project query
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, D]
        
        # Project memories
        k_proj = self.key_proj(used_keys).unsqueeze(0)  # [1, M, D]
        v_proj = self.value_proj(used_values).unsqueeze(0)  # [1, M, D]
        
        # Expand for batch
        batch_size = query.size(0)
        if batch_size > 1:
            k_proj = k_proj.expand(batch_size, -1, -1)
            v_proj = v_proj.expand(batch_size, -1, -1)
        
        # Attention-based retrieval
        memory_out, attn_weights = self.memory_attn(q, k_proj, v_proj)
        
        return memory_out.squeeze(1), attn_weights.squeeze(1)
    
    def forward(self, neural_input, store_memory=False):
        """
        Modulate neural input with associative memory.
        
        Args:
            neural_input: Output from neural layer [batch, neural_dim]
            store_memory: Whether to store this pattern
            
        Returns:
            Modulated output [batch, neural_dim]
        """
        # Optionally store pattern
        if store_memory and self.training:
            # Store batch items independently
            for i in range(neural_input.size(0)):
                self.write_memory(neural_input[i], neural_input[i])
        
        # Read relevant memories
        memory_readout = self.read_memory(neural_input)
        
        if memory_readout is None:
            return neural_input
            
        memory_signal, attn_weights = memory_readout
        
        # Compute memory gate (how much to use memory)
        gate_input = torch.cat([neural_input, memory_signal], dim=-1)
        gate = self.memory_gate(gate_input)
        
        # Apply gated memory
        gated_memory = gate * memory_signal
        
        # Integrate neural and memory signals
        combined = torch.cat([neural_input, gated_memory], dim=-1)
        output = self.integration(combined)
        
        return output


class NeuralNetworkWithAstrocytes(nn.Module):
    """
    Example neural network with astrocyte memory modules.
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=10):
        super().__init__()
        
        # Regular neural layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
        # Astrocyte modules - one per layer
        self.astrocyte1 = AstrocyteMemoryModule(hidden_dim)
        self.astrocyte2 = AstrocyteMemoryModule(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x, use_memory=True, store_memory=False):
        """
        Forward pass with optional memory usage.
        
        Args:
            x: Input tensor (e.g., BART embeddings)
            use_memory: Whether to use astrocyte memory
            store_memory: Whether to store patterns in memory
        """
        # Layer 1 + Astrocyte 1
        x = self.layer1(x)
        x = self.relu(x)
        if use_memory:
            x = self.astrocyte1(x, store_memory=store_memory)
        x = self.dropout(x)
        
        # Layer 2 + Astrocyte 2  
        x = self.layer2(x)
        x = self.relu(x)
        if use_memory:
            x = self.astrocyte2(x, store_memory=store_memory)
        x = self.dropout(x)
        
        # Output layer
        x = self.layer3(x)
        
        return x


def demo_astrocyte_network():
    """Demonstrate the astrocyte-enhanced neural network."""
    
    # Initialize BART for embeddings
    print("Loading BART...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    bart.eval()
    
    # Initialize neural network with astrocytes
    model = NeuralNetworkWithAstrocytes()
    model.train()  # Important for memory storage
    
    # Training examples
    training_texts = [
        "Neurons fire action potentials",
        "Synapses connect neurons together", 
        "Dendrites receive incoming signals",
        "Axons transmit outgoing signals",
        "Neurotransmitters carry chemical signals"
    ]
    
    print("\nStoring patterns in astrocyte memory...")
    for text in training_texts:
        # Get BART embedding
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            embedding = bart(**inputs).last_hidden_state.mean(dim=1)
        
        # Forward pass with memory storage
        output = model(embedding, use_memory=True, store_memory=True)
        print(f"Stored: {text}")
    
    # Test with new inputs
    model.eval()  # Switch to eval mode
    print("\nTesting with new inputs...")
    
    test_texts = [
        "How do neurons communicate?",
        "What are action potentials?",
        "Tell me about synaptic transmission"
    ]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            embedding = bart(**inputs).last_hidden_state.mean(dim=1)
            
            # Compare with and without memory
            output_no_memory = model(embedding, use_memory=False)
            output_with_memory = model(embedding, use_memory=True)
            
            print(f"\nQuery: {text}")
            print(f"Output magnitude without memory: {output_no_memory.norm().item():.3f}")
            print(f"Output magnitude with memory: {output_with_memory.norm().item():.3f}")
            print(f"Difference: {(output_with_memory - output_no_memory).norm().item():.3f}")


def train_example():
    """Example of training a network with astrocyte memory."""
    
    # Simple dataset
    texts = [
        ("The cell uses ATP for energy", 0),  # Class 0: cellular
        ("Neurons transmit electrical signals", 1),  # Class 1: neural
        ("Mitochondria produce ATP", 0),
        ("Synapses connect neurons", 1),
        ("DNA stores genetic information", 0),
        ("Axons carry action potentials", 1)
    ]
    
    # Initialize models
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    bart.eval()
    
    model = NeuralNetworkWithAstrocytes(input_dim=768, hidden_dim=256, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Training with astrocyte memory...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        for text, label in texts:
            # Get embedding
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                embedding = bart(**inputs).last_hidden_state.mean(dim=1)
            
            # Forward pass with memory
            output = model(embedding, use_memory=True, store_memory=True)
            
            # Compute loss
            target = torch.tensor([label])
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(texts):.4f}")
    
    # Test
    print("\nTesting...")
    model.eval()
    
    test_texts = [
        "How do cells produce energy?",  # Should activate cellular memories
        "How do neurons communicate?",   # Should activate neural memories
    ]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            embedding = bart(**inputs).last_hidden_state.mean(dim=1)
            output = model(embedding, use_memory=True)
            probs = F.softmax(output, dim=-1)
            
        print(f"\n'{text}'")
        print(f"Cellular: {probs[0,0]:.3f}, Neural: {probs[0,1]:.3f}")


if __name__ == "__main__":
    print("=== Astrocyte Network Demo ===")
    demo_astrocyte_network()
    
    print("\n\n=== Training Example ===")
    train_example()
