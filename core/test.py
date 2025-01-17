import torch
from embedding_analyzer import EmbeddingAnalyzer

# Initialize EmbeddingAnalyzer
model_path = "bert-base-uncased"  # Change to your desired model
device = "cpu"  # Use "cuda" if you have a GPU
analyzer = EmbeddingAnalyzer(model_path=model_path, device=device)

# Test Input Data
texts = ["Hello world", "How are you?", "AI is fascinating", "The future is bright", "Hello AI"]

# Step 1: Extract Token Embeddings
print("Extracting token embeddings...")
embeddings = analyzer.get_token_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}")  # Should be (len(texts), seq_len, hidden_dim)

# Use average pooling to get sentence-level embeddings
sentence_embeddings = embeddings.mean(dim=1)  # Shape: (len(texts), hidden_dim)
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")

# Step 2: Compute Cosine Similarity
print("Computing cosine similarity...")
similarity = analyzer.compute_cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings)
print(f"Cosine similarity with the first text: {similarity}")

# Step 3: Build and Search Nearest Neighbors
print("Building nearest neighbor index...")
nn_index = analyzer.build_sklearn_index(sentence_embeddings, metric="cosine")
query_embedding = sentence_embeddings[0].unsqueeze(0)
distances, indices = analyzer.search_neighbors(nn_index, query_embedding, k=3)
print(f"Nearest neighbors for '{texts[0]}':")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. '{texts[idx]}' (Distance: {distances[0][i]:.4f})")

# Step 4: Clustering with K-Means
print("Clustering embeddings with K-Means...")
n_clusters = 3
labels = analyzer.kmeans_clustering(sentence_embeddings, n_clusters=n_clusters)
print(f"Cluster labels: {labels}")

# Step 5: Visualize Clusters
print("Visualizing clusters...")
analyzer.visualize_clusters(sentence_embeddings, labels)

print("All tests completed successfully!")
