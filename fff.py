import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Custom fast linear layer
class FastLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_size = in_features
        self.weight = nn.Parameter(F.normalize(torch.randn(out_features, in_features), p=2, dim=-1))
        #self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return x @ self.weight.T #+ self.bias

# FFF model
class FFF(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, leaf_size, depth):
        super().__init__()

        # Tree nodes
        self.nodes = nn.ModuleList()
        for _ in range(depth-1):
            node = FastLinear(input_size, 1)
            self.nodes.append(node)

        # Leaves
        self.leaves = nn.ModuleList()
        for _ in range(2**depth):
            leaf = nn.Sequential(
                FastLinear(input_size, leaf_size),
                #nn.ReLU(),
                FastLinear(leaf_size, num_classes)
            )
            self.leaves.append(leaf)

    def forward(self, x):
        # Soft routing
        decisions = [torch.sigmoid(node(x)) for node in self.nodes]

        # Initialize out tensor with the correct shape
        sample_leaf = self.leaves[0](x)
        out = torch.zeros(sample_leaf.shape[0], 1, sample_leaf.shape[1])

        for i, leaf in enumerate(self.leaves):
            decision = decisions[i % len(decisions)]
            #print(f"before, decision.shape = {decision.shape}")
            #print(f"before, leaf(x).shape = {leaf(x).shape}")
            leaf_x = leaf(x).view(x.size(0), num_classes, -1)
            leaf_x = leaf_x.transpose(1, 2)
            #print(f"after, leaf(x).shape = {leaf_x.shape}")
            decision = decision.unsqueeze(2).expand_as(leaf_x)
            #print(f"after, decision.shape = {decision.shape}")
            out += decision * leaf_x

        #print(f"before, out.shape = {out.shape}")
        # Reduce the out tensor along the second dimension
        out = torch.mean(out, dim=1)  # or torch.sum(out, dim=1)
        #print(f"after, out.shape = {out.shape}")

        return out

    def forward_hard(self, x):
        # Hard routing
        decisions = [torch.sigmoid(node(x)) for node in self.nodes]
        decisions = [torch.round(d) for d in decisions]

        out = []
        for i in range(x.size(0)):  # loop over each item in the batch
            leaf_index = 0
            for d in decisions:
                d_i = d[i].type(torch.long)  # Convert to long tensor
                leaf_index = (leaf_index << 1) | int(d_i.item())  # Convert d_i to int

            out_i = self.leaves[int(leaf_index)](x[i])  # Convert leaf_index to integer
            out.append(out_i)

        # Aggregate the leaf outputs
        outputs = torch.stack(out, dim=0)  # Shape: [64, 49, 10]
        #print(f"outputs.shape = {outputs.shape}")
        outputs = torch.mean(outputs, dim=1)  # Take the average along the second dimension
        #print(f"outputs.shape = {outputs.shape}")

        return outputs


# Transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST dataset
trainset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)

# Data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Hyperparameters
num_epochs = 10
input_size = 28*28
num_classes = 10
hidden_size = 512
leaf_size = 16
depth = 3

# Create model, loss, and optimizer
model = FFF(input_size, num_classes, hidden_size, leaf_size, depth)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
loss = torch.tensor(0)  # initialize total_loss as a tensor
total_loss = torch.tensor(0)  # initialize total_loss as a tensor
h = 0.5  # h is a hyperparameter that controls the impact of the hardening loss

# Training loop
for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        #print(f"A. images.shape = {images.shape}")
        images = images.view(-1, input_size)
        #print(f"B. images.shape = {images.shape}")

        optimizer.zero_grad()

        # Use soft routing
        outputs = model(images)

        #print(f"outputs.shape = {outputs.shape}")
        #print(f"labels.shape = {labels.shape}")
        loss = criterion(outputs, labels)

        # Hardening loss
        decisions = [torch.sigmoid(node(images)) for node in model.nodes]
        decisions = torch.stack(decisions)
        hardening_loss = torch.mean(F.binary_cross_entropy(decisions, torch.zeros_like(decisions)))

        # Combine losses
        total_loss = loss + h * hardening_loss

        # Backpropagate total loss
        total_loss.backward()

        optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss.item()))

# Evaluate on test set
correct = 0
with torch.no_grad():
    for images, labels in testset:
        # Use hard routing
        #outputs = model.forward_hard(images)
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d%%' % (100 * correct / len(testset)))
