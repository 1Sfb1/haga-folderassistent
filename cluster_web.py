import chromadb
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

# 1. Database inladen
print("Database laden...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("haga_folders")

data = collection.get(include=["embeddings", "metadatas"])
titles = [meta.get("title", meta.get("filename", "Onbekend")) for meta in data["metadatas"]]
embeddings = np.array(data["embeddings"])

# Om te voorkomen dat je browser crasht, pakken we een steekproef van ~250 chunks.
# (Pas dit aan als je PC sterk genoeg is!)
limit = 250
sampled_titles = titles[:limit]
sampled_embeddings = embeddings[:limit]

# 2. Cosine Similarity berekenen
print("Wiskunde draaien (Cosine Similarity)...")
similarity_matrix = cosine_similarity(sampled_embeddings)

# 3. Graaf opbouwen
print("Netwerk bouwen...")
G = nx.Graph()

# Nodes (Folders) toevoegen
for i, title in enumerate(sampled_titles):
    # Maak de titel iets korter voor de visuele bolletjes
    short_label = title.replace(".pdf", "").split("-", 1)[-1].capitalize() if "-" in title else title
    short_label = short_label[:25] + "..." if len(short_label) > 25 else short_label
    
    # 'label' is de tekst op het bolletje, 'title' is de hover-tooltip
    G.add_node(i, label=short_label, title=title)

# Edges (Verbindingen) toevoegen met een DREMPELWAARDE
THRESHOLD = 0.88  # <--- HIER KUN JE MEE SPELEN!
# 0.85 geeft meer lijnen, 0.92 geeft alleen super-strikte clusters.

for i in range(len(sampled_titles)):
    for j in range(i + 1, len(sampled_titles)):
        sim = similarity_matrix[i, j]
        if sim > THRESHOLD:
            # Hoe hoger de correlatie, hoe dikker de lijn
            dikte = (sim - THRESHOLD) * 20 
            G.add_edge(i, j, value=dikte, title=f"Similarity: {sim:.2f}")

# Gooi folders weg die nergens op lijken (geïsoleerde nodes) voor een schoner beeld
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
print(f"  > {len(isolates)} eenzame folders verwijderd voor een schoner web.")

# 4. De Interactieve Magie: PyVis
print("Interactieve HTML webpagina genereren...")
# We gebruiken donkere modus voor dat hacker-tech gevoel
net = Network(height="100vh", width="100%", bgcolor="#1a1a1a", font_color="#ffffff")
net.from_nx(G)

# Fysica aanzetten: bolletjes stoten elkaar af, verbonden lijnen trekken als elastiekjes
net.repulsion(
    node_distance=200, 
    central_gravity=0.05, 
    spring_length=150, 
    spring_strength=0.05, 
    damping=0.09
)

# Genereer het HTML bestand
output_file = "haga_cluster_web.html"
net.show(output_file, notebook=False)
print(f"\n✅ Magie is klaar! Open het bestand '{output_file}' dubbelklikkend in je Chrome/Edge browser.")