from flask import Flask, request, jsonify
from embedding_analyzer import EmbeddingAnalyzer
import torch
from flask_cors import CORS

# Initialize Flask app and EmbeddingAnalyzer
app = Flask(__name__)
model_path = "bert-base-uncased"  # Change this if needed
analyzer = EmbeddingAnalyzer(model_path=model_path, device="cpu")
CORS(app)  # Enable CORS for all routes


@app.route("/embeddings", methods=["POST"])
def get_embeddings():
    """
    Extract token embeddings for a list of inputs.
    """
    data = request.json
    inputs = data.get("texts", [])
    try:
        # Generate embeddings and apply mean pooling
        embeddings = analyzer.get_token_embeddings(inputs).mean(dim=1).tolist()
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        return jsonify({"error": f"Failed to get embeddings: {str(e)}"}), 500


@app.route("/cosine_similarity", methods=["POST"])
def cosine_similarity():
    """
    Compute cosine similarity between two texts.
    """
    data = request.json
    texts = data.get("texts", [])
    if len(texts) != 2:
        return jsonify({"error": "Provide exactly two texts."}), 400

    try:
        # Generate embeddings for both texts
        embeddings = analyzer.get_token_embeddings(texts)  # Shape: (2, embedding_dim)

        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = analyzer.compute_cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ).item()

        return jsonify({"cosine_similarity": similarity})
    except Exception as e:
        return jsonify({"error": f"Failed to compute cosine similarity: {str(e)}"}), 500



@app.route("/nearest_neighbors", methods=["POST"])
def nearest_neighbors():
    """
    Find nearest neighbors for a query text within a dataset of texts.
    """
    data = request.json
    texts = data.get("texts", [])
    query_text = data.get("query", "")
    k = data.get("k", 3)

    if not texts or not query_text:
        return jsonify({"error": "Provide both dataset texts and a query text."}), 400

    try:
        # Generate embeddings for the dataset and query
        embeddings = analyzer.get_token_embeddings(texts)  # Shape: (num_texts, embedding_dim)
        query_embedding = analyzer.get_token_embeddings([query_text])  # Shape: (1, embedding_dim)

        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        # Adjust k to not exceed dataset size
        k = min(k, len(texts))

        # Build the nearest neighbor index and search
        index = analyzer.build_sklearn_index(embeddings.cpu(), metric="cosine")
        distances, indices = analyzer.search_neighbors(index, query_embedding.cpu(), k=k)

        return jsonify({"distances": distances.tolist(), "indices": indices.tolist()})
    except Exception as e:
        return jsonify({"error": f"Failed to find nearest neighbors: {str(e)}"}), 500


@app.route("/clustering", methods=["POST"])
def clustering():
    """
    Perform K-Means clustering on embeddings.
    """
    data = request.json
    embeddings = data.get("embeddings", [])
    n_clusters = data.get("n_clusters", 3)

    if not embeddings:
        return jsonify({"error": "Provide embeddings for clustering."}), 400

    try:
        # Convert embeddings to tensor and perform clustering
        embeddings_tensor = torch.tensor(embeddings)
        labels = analyzer.kmeans_clustering(embeddings_tensor, n_clusters=n_clusters)
        return jsonify({"cluster_labels": labels})
    except Exception as e:
        return jsonify({"error": f"Failed to perform clustering: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
