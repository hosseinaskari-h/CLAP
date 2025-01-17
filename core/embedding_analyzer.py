from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn.functional import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class EmbeddingAnalyzer:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the embedding analyzer.
        :param model_path: Path or name of the pre-trained model.
        :param device: Device to run the model on ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

    def get_token_embeddings(self, inputs: list, batch_size: int = 16) -> torch.Tensor:
        """
        Extract token embeddings for a list of inputs.
        :param inputs: List of input texts.
        :param batch_size: Number of inputs to process at once.
        :return: Tensor of token embeddings (batch_size, hidden_dim).
        """
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

            # Apply mean pooling to get sequence-level embeddings
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_pooled_embeddings)

        # Concatenate all embeddings (if batching)
        return torch.cat(all_embeddings, dim=0)


    def compute_cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two embeddings.
        :param emb1: First embedding tensor (1D or 2D).
        :param emb2: Second embedding tensor (1D or 2D).
        :return: Cosine similarity score(s).
        """
        # Normalize embeddings
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        return similarity

    def build_sklearn_index(self, dataset: torch.Tensor, metric: str = "cosine") -> NearestNeighbors:
        """
        Build a nearest neighbor index using Scikit-learn.
        :param dataset: Dataset of embeddings (Tensor or numpy array).
        :param metric: Distance metric for nearest neighbors (e.g., "euclidean", "cosine").
        :return: Fitted NearestNeighbors object.
        """
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.cpu().numpy()
        nn = NearestNeighbors(metric=metric)
        nn.fit(dataset)
        return nn

    def search_neighbors(self, index: NearestNeighbors, query: torch.Tensor, k: int = 5) -> tuple:
        """
        Search for nearest neighbors using Scikit-learn.
        :param index: NearestNeighbors object.
        :param query: Query embedding(s).
        :param k: Number of neighbors to return.
        :return: Distances and indices of neighbors.
        """
        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()
        distances, indices = index.kneighbors(query, n_neighbors=k)
        return distances, indices

    def kmeans_clustering(self, embeddings: torch.Tensor, n_clusters: int = 5) -> list:
        """
        Cluster embeddings using K-means.
        :param embeddings: Embeddings to cluster.
        :param n_clusters: Number of clusters.
        :return: Cluster labels for each embedding.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels

    def visualize_clusters(self, embeddings: torch.Tensor, labels: list, n_components: int = 2):
        """
        Visualize clustering results.
        :param embeddings: Embeddings to visualize.
        :param labels: Cluster labels for each embedding.
        :param n_components: Number of PCA dimensions for visualization.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap="viridis",
            s=50,
        )
        plt.colorbar(scatter)
        plt.title("Cluster Visualization")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()
