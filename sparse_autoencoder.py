import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# outputs holder
outputs = ["lebron james plays basketball"]

# word2vec
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

def encode_texts(texts):
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model_bert(**encoded_inputs).last_hidden_state.mean(dim=1)
    return embeddings

embeddings = encode_texts(outputs)

# data set and data loading
dataset = TensorDataset(embeddings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# SAE implementation
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_lambda=1e-3):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        sparsity_loss = self.sparsity_lambda * torch.sum(torch.abs(z))
        x_recon = self.decoder(z)
        return x_recon, sparsity_loss

input_size = embeddings.shape[1]
hidden_size = 64  # CHANGE LATER FOR ACTUAL DATASET
model = SparseAutoencoder(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        inputs = data[0]
        optimizer.zero_grad()
        outputs, sparsity_loss = model(inputs)
        reconstruction_loss = criterion(outputs, inputs)
        loss = reconstruction_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

with torch.no_grad():
    embeddings = embeddings
    latent_representations = torch.relu(model.encoder(embeddings))

# clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(latent_representations)
labels = kmeans.labels_

# cluster analysis
for i in range(2):
    indices = (labels == i).nonzero().flatten()
    print(f"\nCluster {i}:")
    for idx in indices:
        print(outputs[idx])

# reconstruct inputs from specific latent vectors
latent_vector = latent_representations[0]
reconstructed_embedding = model.decoder(latent_vector)
# IMPLEMENT DECODER TO OUTPUT REPRESENTATIONS AS TEXT (GRADED PORTION)