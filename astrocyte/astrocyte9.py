import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer
import numpy as np


class AstrocyteAssociativeMemory(nn.Module):
    """
    An associative memory module inspired by astrocytes that works alongside neurons.
    
    Think of this like a librarian (astrocyte) that helps neurons find related memories
    based on semantic similarity rather than exact matches.
    """
    
    def __init__(self, memory_size=1000, embedding_dim=768, threshold=0.2):
        """
        Args:
            memory_size: Maximum number of memories to store
            embedding_dim: Dimension of BART embeddings (768 for base model)
            threshold: Cosine similarity threshold for memory retrieval
        """
        super().__init__()
        
        # Memory bank - like the astrocyte's "storage cabinet"
        self.memory_bank = nn.Parameter(torch.zeros(memory_size, embedding_dim), 
                                       requires_grad=False)
        self.memory_values = nn.Parameter(torch.zeros(memory_size, embedding_dim), 
                                         requires_grad=False)
        
        # Track which memory slots are used
        self.memory_usage = nn.Parameter(torch.zeros(memory_size, dtype=torch.bool), 
                                        requires_grad=False)
        
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.current_idx = 0
        
        # Simple attention mechanism for memory influence on neurons
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim)
        
        # Gating mechanism - how much astrocyte influences the neuron
        self.influence_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
    def store_memory(self, key_embedding, value_embedding):
        """
        Store a new memory association.
        
        Analogy: Like an astrocyte creating a new synaptic connection pattern.
        """
        with torch.no_grad():
            # Ensure embeddings are 1D
            if key_embedding.dim() > 1:
                key_embedding = key_embedding.squeeze()
            if value_embedding.dim() > 1:
                value_embedding = value_embedding.squeeze()
                
            # Circular buffer storage
            idx = self.current_idx % self.memory_size
            
            self.memory_bank[idx] = key_embedding
            self.memory_values[idx] = value_embedding
            self.memory_usage[idx] = True
            
            self.current_idx += 1
            
    def retrieve_memories(self, query_embedding, top_k=5):
        """
        Retrieve memories based on cosine similarity.
        
        Analogy: Like an astrocyte activating when it recognizes a familiar pattern.
        """
        # Only look at used memory slots
        used_indices = torch.where(self.memory_usage)[0]
        
        if len(used_indices) == 0:
            return None, None
        
        # Ensure query is 1D
        if query_embedding.dim() > 1:
            query_embedding = query_embedding.squeeze()
        
        used_keys = self.memory_bank[used_indices]  # Use keys for similarity!
        used_values = self.memory_values[used_indices]
        
        # Compute cosine similarities with the keys
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(used_keys, dim=-1)
        
        similarities = torch.matmul(query_norm, keys_norm.T).squeeze(0)
        
        # Get top-k memories (always return top k, regardless of threshold)
        top_k = min(top_k, len(similarities))
        values, indices = torch.topk(similarities, top_k)
        
        # Return the corresponding values and similarity scores
        return used_values[indices], values
        
    def modulate_neural_activity(self, neural_output, context_embedding):
        """
        Modulate neural activity based on associative memories.
        
        Analogy: Like astrocytes releasing gliotransmitters to influence nearby neurons.
        """
        # Retrieve relevant memories
        memories, similarities = self.retrieve_memories(context_embedding)
        
        if memories is None or len(memories) == 0:
            return neural_output
        
        # Simple scaled dot-product attention
        # Transform neural output to query
        query = self.query_transform(neural_output)  # [batch_size, hidden_dim]
        
        # Transform memories to keys and values
        keys = self.key_transform(memories)  # [num_memories, hidden_dim]
        values = self.value_transform(memories)  # [num_memories, hidden_dim]
        
        # Compute attention scores
        # query: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        # keys: [num_memories, hidden_dim] -> [1, num_memories, hidden_dim]
        query = query.unsqueeze(1)
        keys = keys.unsqueeze(0)
        
        # Attention scores: [batch_size, 1, num_memories]
        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        
        # Weight by similarities (astrocyte-specific behavior)
        if similarities is not None:
            similarity_weights = similarities.unsqueeze(0).unsqueeze(0)  # [1, 1, num_memories]
            scores = scores * similarity_weights
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, num_memories]
        
        # Apply attention to values
        values = values.unsqueeze(0)  # [1, num_memories, hidden_dim]
        attended_memory = torch.matmul(attention_weights, values)  # [batch_size, 1, hidden_dim]
        attended_memory = attended_memory.squeeze(1)  # [batch_size, hidden_dim]
        
        # Compute influence gate
        combined = torch.cat([neural_output, attended_memory], dim=-1)
        gate = self.influence_gate(combined)
        
        # Modulated output
        modulated_output = neural_output + gate * attended_memory
        
        return modulated_output


class NeuronAstrocyteNetwork(nn.Module):
    """
    A neural network that integrates astrocyte-like associative memory.
    
    Example usage combining BART embeddings with a task-specific neural network.
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=10):
        super().__init__()
        
        # Traditional neural layers
        self.neural_pathway = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Projection layers to match dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.context_projection = nn.Linear(input_dim, hidden_dim)
        
        # Astrocyte memory system
        self.astrocyte_memory = AstrocyteAssociativeMemory(
            memory_size=1000,
            embedding_dim=hidden_dim  # Now matches neural pathway output
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, bart_embedding, store_memory=False):
        """
        Forward pass with astrocyte modulation.
        """
        # Neural processing
        neural_output = self.neural_pathway(bart_embedding)
        
        # Project BART embedding to hidden dimension for context
        context_embedding = self.context_projection(bart_embedding)
        
        # Astrocyte modulation - pass the first item if batched
        if context_embedding.dim() == 2 and context_embedding.size(0) == 1:
            context_for_retrieval = context_embedding[0]
        else:
            context_for_retrieval = context_embedding
            
        modulated_output = self.astrocyte_memory.modulate_neural_activity(
            neural_output, context_for_retrieval
        )
        
        # Store this pattern if learning
        if store_memory:
            # Project input to hidden dim for storage
            key_embedding = self.input_projection(bart_embedding)
            # Store the first item if batched
            if key_embedding.dim() == 2 and key_embedding.size(0) == 1:
                self.astrocyte_memory.store_memory(key_embedding[0], neural_output[0])
            else:
                self.astrocyte_memory.store_memory(key_embedding, neural_output)
        
        # Final output
        output = self.output_layer(modulated_output)
        
        return output


# Example usage
def example_usage():
    """
    Demonstrates how to use the astrocyte-inspired memory system.
    """
    # Initialize BART for embeddings
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    
    # Initialize our network
    network = NeuronAstrocyteNetwork()
    
    # Example: Process and store memories
    texts = [
        "The mitochondria is the powerhouse of the cell",
        "Neurons communicate through synapses",
        "Astrocytes support neuronal function"
    ]
    
    # Training phase - store memories
    for text in texts:
        # Get BART embeddings
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            bart_output = bart(**inputs)
            # Use pooled output (average of last hidden states)
            embedding = bart_output.last_hidden_state.mean(dim=1)
        
        # Process with memory storage
        output = network(embedding, store_memory=True)
        print(f"Processed and stored: {text[:30]}...")
    
    # Inference phase - use associative memories
    query_text = "Glial cells help neurons"
    inputs = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        bart_output = bart(**inputs)
        query_embedding = bart_output.last_hidden_state.mean(dim=1)
        
        # Process with memory retrieval
        output = network(query_embedding, store_memory=False)
        
        # Check what memories were activated
        # Use the projected embedding for memory retrieval check
        context_embedding = network.context_projection(query_embedding)
        memories, similarities = network.astrocyte_memory.retrieve_memories(context_embedding)
        if memories is not None:
            print(f"\nQuery: {query_text}")
            print(f"Retrieved {len(memories)} similar memories")
            print(f"Similarity scores: {similarities.tolist()}")


# Advanced example: Memory consolidation
class MemoryConsolidation(nn.Module):
    """
    Implements memory consolidation inspired by astrocyte-neuron interactions
    during sleep/rest periods.
    """
    
    def __init__(self, astrocyte_memory):
        super().__init__()
        self.astrocyte_memory = astrocyte_memory
        
        # Consolidation network
        self.consolidation = nn.Sequential(
            nn.Linear(astrocyte_memory.embedding_dim * 2, 
                     astrocyte_memory.embedding_dim),
            nn.ReLU(),
            nn.Linear(astrocyte_memory.embedding_dim, 
                     astrocyte_memory.embedding_dim)
        )
        
    def consolidate_memories(self, num_iterations=10):
        """
        Consolidate memories by finding and strengthening associations.
        
        Analogy: Like astrocytes during sleep, reorganizing synaptic connections.
        """
        used_indices = torch.where(self.astrocyte_memory.memory_usage)[0]
        
        if len(used_indices) < 2:
            return
        
        for _ in range(num_iterations):
            # Randomly select pairs of memories
            idx1, idx2 = torch.randint(0, len(used_indices), (2,))
            
            mem1 = self.astrocyte_memory.memory_bank[used_indices[idx1]]
            mem2 = self.astrocyte_memory.memory_bank[used_indices[idx2]]
            
            # Compute association strength
            similarity = F.cosine_similarity(mem1, mem2, dim=0)
            
            if similarity > 0.5:  # If memories are related
                # Create consolidated representation
                combined = torch.cat([mem1, mem2])
                consolidated = self.consolidation(combined)
                
                # Store as new memory
                self.astrocyte_memory.store_memory(consolidated, consolidated)


def interactive_dialog():
    """
    Interactive dialog system showing astrocyte memory in action.
    """
    print("Initializing BART and Astrocyte-Neural Network...")
    
    # Initialize models
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    bart.eval()  # Set to evaluation mode
    
    # Initialize our network
    network = NeuronAstrocyteNetwork()
    
    # Initial knowledge to store
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
    
    # Store initial memories
    for i, text in enumerate(initial_knowledge):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            bart_output = bart(**inputs)
            embedding = bart_output.last_hidden_state.mean(dim=1)
        
        # Process and store
        output = network(embedding, store_memory=True)
        print(f"[{i+1}/10] Stored: {text[:50]}...")
    
    # Debug: Check what's actually stored
    print(f"\nDebug: Total memories stored: {network.astrocyte_memory.current_idx}")
    print(f"Memory usage sum: {network.astrocyte_memory.memory_usage.sum().item()}")
    
    # Save the model
    print("\nSaving model...")
    torch.save({
        'network_state_dict': network.state_dict(),
        'memory_bank': network.astrocyte_memory.memory_bank,
        'memory_values': network.astrocyte_memory.memory_values,
        'memory_usage': network.astrocyte_memory.memory_usage,
        'current_idx': network.astrocyte_memory.current_idx
    }, 'astrocyte_neural_model.pth')
    print("Model saved to 'astrocyte_neural_model.pth'")
    
    print("\n" + "="*60)
    print("INTERACTIVE DIALOG - Astrocyte Associative Memory Demo")
    print("="*60)
    print("\nType your questions about cells, neurons, or neuroscience.")
    print("Type 'quit' to exit, 'memories' to see stored count.")
    print("-"*60 + "\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'memories':
            count = network.astrocyte_memory.memory_usage.sum().item()
            print(f"\nSystem: Currently storing {count} memories")
            continue
        elif not user_input:
            continue
        
        # Process the query
        inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            # Get BART embedding
            bart_output = bart(**inputs)
            query_embedding = bart_output.last_hidden_state.mean(dim=1)
            
            # Get the context embedding for memory retrieval
            context_embedding = network.context_projection(query_embedding)
            
            # Retrieve similar memories BEFORE processing
            memories, similarities = network.astrocyte_memory.retrieve_memories(
                context_embedding, top_k=5
            )
            
            # Process through network
            output = network(query_embedding, store_memory=False)
            
            # Show results
            print(f"\nSystem Analysis:")
            print("-" * 40)
            
            if memories is not None and len(memories) > 0:
                print(f"Found {len(memories)} relevant memories:")
                for i, (memory, sim) in enumerate(zip(memories, similarities)):
                    print(f"  Memory {i+1}: Similarity = {sim.item():.3f}")
                    
                    # Find which original text this corresponds to
                    # We need to find the matching key in memory_bank
                    best_match_idx = None
                    best_match_sim = -1
                    
                    for j in range(min(network.astrocyte_memory.current_idx, len(initial_knowledge))):
                        stored_value = network.astrocyte_memory.memory_values[j]
                        # Ensure both tensors are 1D for comparison
                        memory_flat = memory.flatten()
                        stored_flat = stored_value.flatten()
                        
                        # Compare with the retrieved memory value
                        value_sim = F.cosine_similarity(memory_flat, stored_flat, dim=0).item()
                        if value_sim > best_match_sim:
                            best_match_sim = value_sim
                            best_match_idx = j
                    
                    if best_match_idx is not None and best_match_sim > 0.95:
                        print(f"    → Related to: \"{initial_knowledge[best_match_idx][:60]}...\"")
                
                # Show if these are below threshold
                threshold_note = ""
                if similarities[0].item() < network.astrocyte_memory.threshold:
                    threshold_note = f" (below threshold {network.astrocyte_memory.threshold})"
                print(f"\nClosest matches found{threshold_note}")
            else:
                print("No memories found in the system")
            
            # Show neural output characteristics
            output_magnitude = output.norm().item()
            print(f"\nNeural output magnitude: {output_magnitude:.3f}")
            print(f"Output shape: {output.shape}")
            
            # Optionally store this as a new memory
            if len(user_input.split()) > 5:  # Only store substantial inputs
                store = input("\nStore this as new memory? (y/n): ").lower() == 'y'
                if store:
                    network(query_embedding, store_memory=True)
                    print("Memory stored!")


def load_and_query():
    """
    Load a saved model and query it.
    """
    print("Loading saved model...")
    
    # Initialize models
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    network = NeuronAstrocyteNetwork()
    
    # Load saved state
    checkpoint = torch.load('astrocyte_neural_model.pth')
    network.load_state_dict(checkpoint['network_state_dict'])
    network.astrocyte_memory.memory_bank.data = checkpoint['memory_bank']
    network.astrocyte_memory.memory_values.data = checkpoint['memory_values']
    network.astrocyte_memory.memory_usage.data = checkpoint['memory_usage']
    network.astrocyte_memory.current_idx = checkpoint['current_idx']
    
    print("Model loaded successfully!")
    
    # Test query
    test_query = "How do brain cells communicate?"
    print(f"\nTest query: '{test_query}'")
    
    inputs = tokenizer(test_query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        bart_output = bart(**inputs)
        embedding = bart_output.last_hidden_state.mean(dim=1)
        
        context_embedding = network.context_projection(embedding)
        memories, similarities = network.astrocyte_memory.retrieve_memories(context_embedding)
        
        if memories is not None:
            print(f"Retrieved {len(memories)} memories with similarities: {similarities.tolist()}")


if __name__ == "__main__":
    # Run interactive dialog
    interactive_dialog()
    
    # Uncomment to test loading
    # load_and_query()
