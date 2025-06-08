import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer
import numpy as np


class SimpleAstrocyteMemory:
    """
    Simple associative memory using BART embeddings.
    No fancy stuff - just store and retrieve based on cosine similarity.
    """
    
    def __init__(self):
        self.memory_texts = []
        self.memory_embeddings = []
        
    def store(self, text, embedding):
        """Store text and its embedding."""
        self.memory_texts.append(text)
        # Clone and detach to avoid gradient issues
        self.memory_embeddings.append(embedding.clone().detach().cpu())
        
    def retrieve(self, query_embedding, top_k=5):
        """Find most similar memories."""
        if not self.memory_embeddings:
            return [], []
        
        # Move all to same device
        device = query_embedding.device
        memory_matrix = torch.stack(self.memory_embeddings).to(device)
        
        # Normalize for cosine similarity
        query_norm = F.normalize(query_embedding.view(1, -1), p=2, dim=1)
        memory_norm = F.normalize(memory_matrix, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_norm, memory_norm.t()).squeeze(0)
        
        # Get top k
        k = min(top_k, len(similarities))
        top_sims, top_indices = torch.topk(similarities, k)
        
        results = []
        for idx in top_indices:
            results.append(self.memory_texts[idx.item()])
            
        return results, top_sims


def test_astrocyte_memory():
    """Test the memory system with BART."""
    print("Initializing BART...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    bart.eval()
    
    # Create memory
    memory = SimpleAstrocyteMemory()
    
    # Knowledge base
    knowledge = [
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
    
    print("\nStoring knowledge...")
    for i, text in enumerate(knowledge):
        # Get BART embedding
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bart(**inputs)
            # Use the mean of encoder's last hidden state
            embedding = outputs.encoder_last_hidden_state.mean(dim=1).squeeze(0)
        
        memory.store(text, embedding)
        print(f"Stored {i+1}: {text[:50]}...")
    
    print("\n" + "="*60)
    print("Testing retrieval...")
    print("="*60)
    
    # Test queries
    test_queries = [
        "what do axons do?",
        "tell me about mitochondria",
        "how do neurons communicate?",
        "what protects the brain?",
        "what happens during sleep?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Get query embedding
        inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bart(**inputs)
            query_embedding = outputs.encoder_last_hidden_state.mean(dim=1).squeeze(0)
        
        # Retrieve
        results, similarities = memory.retrieve(query_embedding, top_k=3)
        
        for i, (text, sim) in enumerate(zip(results, similarities)):
            print(f"{i+1}. Similarity: {sim:.3f}")
            print(f"   → {text}")


def interactive_simple():
    """Simple interactive version."""
    print("Initializing BART...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart = BartModel.from_pretrained('facebook/bart-base')
    bart.eval()
    
    memory = SimpleAstrocyteMemory()
    
    # Knowledge base
    knowledge = [
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
    
    print("\nStoring knowledge...")
    for text in knowledge:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bart(**inputs)
            embedding = outputs.encoder_last_hidden_state.mean(dim=1).squeeze(0)
        memory.store(text, embedding)
    
    print(f"\nStored {len(knowledge)} memories.")
    print("\nType 'quit' to exit.")
    print("-" * 60)
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
            
        # Get embedding
        inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bart(**inputs)
            query_embedding = outputs.encoder_last_hidden_state.mean(dim=1).squeeze(0)
        
        # Retrieve
        results, similarities = memory.retrieve(query_embedding, top_k=5)
        
        print("\nRelevant memories:")
        print("-" * 40)
        for i, (text, sim) in enumerate(zip(results, similarities)):
            print(f"{i+1}. Similarity: {sim:.3f}")
            print(f"   → {text[:70]}...")


if __name__ == "__main__":
    # Run the test first to see if it works
    test_astrocyte_memory()
    
    # Then run interactive
    print("\n\n" + "="*60)
    interactive_simple()
