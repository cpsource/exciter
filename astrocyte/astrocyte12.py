import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np


class AstrocyteAssociativeMemory(nn.Module):
    """
    An associative memory module inspired by astrocytes that works alongside neurons.
    Uses Sentence-BERT for proper semantic similarity.
    """
    
    def __init__(self, memory_size=1000, embedding_dim=384, threshold=0.3):
        super().__init__()
        
        # Memory storage
        self.memory_keys = []  # List to store embeddings
        self.memory_texts = []  # List to store original texts
        self.memory_values = []  # List to store neural outputs
        
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        
        # Simple attention mechanism
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim)
        
        # Gating mechanism
        self.influence_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
    def store_memory(self, key_embedding, text, value_embedding):
        """Store a new memory."""
        if len(self.memory_keys) >= self.memory_size:
            # Remove oldest memory
            self.memory_keys.pop(0)
            self.memory_texts.pop(0)
            self.memory_values.pop(0)
            
        self.memory_keys.append(key_embedding.detach().cpu())
        self.memory_texts.append(text)
        self.memory_values.append(value_embedding.detach())
        
    def retrieve_memories(self, query_embedding, top_k=5):
        """Retrieve memories based on cosine similarity."""
        if len(self.memory_keys) == 0:
            return None, None, None
        
        # Convert to tensors
        keys_tensor = torch.stack(self.memory_keys).to(query_embedding.device)
        values_tensor = torch.stack(self.memory_values).to(query_embedding.device)
        
        # Compute cosine similarities
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(keys_tensor, dim=-1)
        
        similarities = torch.matmul(query_norm, keys_norm.T).squeeze(0)
        
        # Get top-k
        top_k = min(top_k, len(similarities))
        values, indices = torch.topk(similarities, top_k)
        
        top_texts = [self.memory_texts[idx] for idx in indices]
        
        return values_tensor[indices], values, top_texts
        
    def modulate_neural_activity(self, neural_output, query_embedding):
        """Modulate neural activity based on associative memories."""
        memories, similarities, texts = self.retrieve_memories(query_embedding)
        
        if memories is None or len(memories) == 0:
            return neural_output
        
        # Simple attention-based combination
        # Weight memories by their similarities
        weights = F.softmax(similarities, dim=0)
        weighted_memory = torch.sum(memories * weights.unsqueeze(-1), dim=0)
        
        # Compute influence gate
        combined = torch.cat([neural_output, weighted_memory], dim=-1)
        gate = self.influence_gate(combined)
        
        # Modulated output
        modulated_output = neural_output + gate * weighted_memory
        
        return modulated_output


class NeuronAstrocyteNetwork(nn.Module):
    """Neural network with astrocyte-like associative memory."""
    
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=10):
        super().__init__()
        
        # Neural pathway
        self.neural_pathway = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Astrocyte memory
        self.astrocyte_memory = AstrocyteAssociativeMemory(
            memory_size=1000,
            embedding_dim=hidden_dim
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, embedding, text=None, store_memory=False):
        """Forward pass."""
        # Neural processing
        neural_output = self.neural_pathway(embedding)
        
        # Astrocyte modulation
        modulated_output = self.astrocyte_memory.modulate_neural_activity(
            neural_output, neural_output  # Use neural output as query
        )
        
        # Store if requested
        if store_memory and text is not None:
            self.astrocyte_memory.store_memory(embedding, text, neural_output)
        
        # Final output
        output = self.output_layer(modulated_output)
        
        return output


def interactive_dialog():
    """Interactive dialog with working semantic similarity."""
    print("Initializing Sentence-BERT and Astrocyte-Neural Network...")
    
    # Use sentence-transformers for proper semantic embeddings
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good
    
    # Initialize network
    network = NeuronAstrocyteNetwork(input_dim=384, hidden_dim=256)
    
    # Initial knowledge
    initial_knowledge = [
        "The mitochondria is the powerhouse of the cell",
        "Neurons communicate through synapses using neurotransmitters",
        "Astrocytes support neuronal function and maintain homeostasis",
        "Dendrites receive signals from other neurons",
        "Axons transmit electrical signals called action potentials",
        "Glial cells provide support and protection for neurons",
        "Synaptic plasticity is the basis of learning and memory",
        "The blood-brain barrier protects the brain from harmful substances",
        "Neurotransmitters like dopamine and serotonin regulate mood",
        "Memory consolidation occurs during sleep"
    ]
    
    print("\nStoring initial knowledge base...")
    print("-" * 50)
    
    # Store memories
    for i, text in enumerate(initial_knowledge):
        embedding = encoder.encode(text, convert_to_tensor=True)
        output = network(embedding, text=text, store_memory=True)
        print(f"[{i+1}/10] Stored: {text[:50]}...")
    
    print("\n" + "="*60)
    print("INTERACTIVE DIALOG - Working Astrocyte Memory")
    print("="*60)
    print("\nAsk questions about cells, neurons, or neuroscience.")
    print("Type 'quit' to exit.")
    print("-"*60 + "\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif not user_input:
            continue
        
        # Get embedding
        query_embedding = encoder.encode(user_input, convert_to_tensor=True)
        
        # Get similar memories directly using sentence embeddings
        memory_embeddings = torch.stack(network.astrocyte_memory.memory_keys).to(query_embedding.device)
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        memory_norm = F.normalize(memory_embeddings, dim=-1)
        similarities = torch.matmul(query_norm, memory_norm.T).squeeze(0)
        
        # Get top 5
        top_values, top_indices = torch.topk(similarities, min(5, len(similarities)))
        
        print(f"\nSystem Analysis:")
        print("-" * 40)
        print(f"Found {len(top_values)} relevant memories:")
        
        for i, (sim, idx) in enumerate(zip(top_values, top_indices)):
            text = network.astrocyte_memory.memory_texts[idx]
            print(f"  Memory {i+1}: Similarity = {sim.item():.3f}")
            print(f"    → \"{text[:60]}...\"")
        
        # Process through network
        output = network(query_embedding)
        print(f"\nNeural output magnitude: {output.norm().item():.3f}")
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    interactive_dialog()
