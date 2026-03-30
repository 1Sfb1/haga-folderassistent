import chromadb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. Database inladen
print("Database laden...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("haga_folders")

# Haal de data op
data = collection.get(include=["embeddings", "metadatas"])

titles = [meta.get("title", meta.get("filename", "Onbekend")) for meta in data["metadatas"]]
embeddings = data["embeddings"]

print(f"Gevonden: {len(titles)} chunks.")

# 2. Pure Python sortering (De fix voor de ValueError!)
# We knopen de titels en vectoren aan elkaar als tuples, sorteren op titel, 
# en pakken er veilig een steekproef uit.
paired_data = sorted(zip(titles, embeddings), key=lambda x: x[0])

# 3. Pak ~60 gelijkmatig verspreide chunks
step = max(1, len(paired_data) // 60)
sampled_data = paired_data[::step][:60]

# Splits ze weer op in twee schone, gesorteerde lijsten
sampled_titles = [item[0] for item in sampled_data]
embeddings_matrix = np.array([item[1] for item in sampled_data])

# 4. Wiskunde: Cosine Similarity Matrix berekenen (S = X * X^T)
print("Cosine Similarity berekenen...")
similarity_matrix = cosine_similarity(embeddings_matrix)

# 5. De visualisatie (Het paradestukje)
plt.figure(figsize=(12, 10))

# We maken labels korter zodat ze op de as passen
short_labels = [str(lbl)[:25] + "..." if len(str(lbl)) > 25 else str(lbl) for lbl in sampled_titles]

# Seaborn heatmap
sns.heatmap(
    similarity_matrix,
    cmap="YlGnBu",          # Geel -> Groen -> Blauw kleurenschaal
    xticklabels=short_labels,
    yticklabels=short_labels,
    linewidths=0.5,
    cbar_kws={'label': 'Cosine Similarity (1.0 = Exacte match)'}
)

# Esthetiek
plt.title("Semantische Correlatie Matrix van HagaZiekenhuis Folders", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()

print("Heatmap gegenereerd! Let op de donkerblauwe vierkanten op de diagonaal.")
plt.show()