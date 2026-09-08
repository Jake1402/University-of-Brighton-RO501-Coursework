import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import os, time

from LoadTrainingSet import ReturnDataset
#from SmallRes import ConvolutionalAgent
from EffecientRes import ConvolutionalAgent

# Custom horizontal flip for continuous action space (apply entire batch)
class RegressiveHFlip(torch.nn.Module):
    def __init__(self):
        super(RegressiveHFlip, self).__init__()
        self.flip = v2.RandomHorizontalFlip(p=1)
    def forward(self, img, label):
        return self.flip(img), -label

def L1Regularisation(model, lambda_=0.001):
    return lambda_ * sum(p.abs().sum() for n, p in model.named_parameters() if "weight" in n and "bn" not in n)

# Standard pytorch training loop
def train(Agent, loss_fn, optim, dataloader, transform):
    total_loss = []
    print_num = (len(dataloader.dataset)//dataloader.batch_size) // 5
    Agent.train(True)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        data, target = transform(data, target)
        optim.zero_grad()
        output = Agent.forward(data)

        loss = loss_fn(output, target.unsqueeze(-1)) + L1Regularisation(Agent, 0.0001)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.parameters(), 1.5)
        optim.step()

        if (batch_idx + 1) % print_num == 0:
            total_loss.append(loss.item())
            print(f"Current Epoch: {epoch+1}, Batch IDX: {batch_idx} Loss: {loss.item():.4f}")
    return total_loss

# Standard pytorch testing loop
@torch.no_grad()
def test(Agent, loss_fn, dataloader, transform):
    total_loss = []
    print_num = (len(dataloader.dataset)//dataloader.batch_size) // 5
    test_loss = 0
    total = 0
    Agent.train(False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = transform(data.to(device), target.to(device))
        output = Agent.forward(data)
        loss = loss_fn(output, target.unsqueeze(-1)).item()
        total += data.size(0)
        test_loss += (loss)*data.size(0)
        if (batch_idx + 1) % print_num == 0:
            total_loss.append(loss)
    avg_loss = test_loss / total
    print(f"Average Loss for entire set: {avg_loss:.4f}")
    return total_loss

if __name__ == "__main__":

    BINS = 31
    MAX_PER_BIN = 550

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Agent = ConvolutionalAgent(action_space=1).to(device)
    print(f"Model created, number of parameters: {sum(param.numel() for param in Agent.parameters() if param.requires_grad)}")
    #sum([torch.abs(param) for param in Agent.parameters() if param.names])
    # Basic hyperparameters, recommend changing with grid search
    lr = 0.0007
    epochs = 20
    split = 0.95
    batch_size = 32

    loss_fn = torch.nn.SmoothL1Loss()
    optim = torch.optim.AdamW(params=Agent.parameters(), lr=lr, weight_decay=0.005)
    #step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs, eta_min=lr/100)

    # Transforms used, basic_transform is just for testing
    train_transform = v2.Compose([
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),     # Normalise between -1 and 1
        v2.RandomApply([RegressiveHFlip()], p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05))
    ])
    basic_transform = v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    # Return datasets and build dataloaders
    TrainDS, TestDS = ReturnDataset("npysets", split, shuffle=True, balance=(BINS, MAX_PER_BIN), figure="Dists")
    TrainDL = DataLoader(TrainDS, batch_size=batch_size, shuffle=True)
    TestsDL = DataLoader(TestDS,  batch_size=batch_size)

    print("Dataloader loaded successfully...")
    print(f"Training Dataset Size: {len(TrainDL.dataset)}")
    print(f"Testing  Dataset Size: {len(TestsDL.dataset)}")

    # Training and track losses for plots
    track_test_loss  = []
    track_train_loss = []
    for epoch in range(epochs):
        init_t = time.time()
        track_train_loss += [sum(train(Agent, loss_fn, optim, TrainDL, train_transform))/5]
        track_test_loss += [sum(test(Agent, loss_fn, TestsDL, basic_transform))/5]
        #step_lr.step()
        torch.save(Agent.state_dict(), os.path.abspath(f"Models\\Model at epoch - {epoch+1}.pt"))
        print(f"Complete time for training process - {time.time() - init_t}")

    plt.plot(track_train_loss)
    plt.plot(track_test_loss)
    plt.title("Train and Test Loss over epochs")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel(f"Iterations - Total Epochs {epochs}")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig(os.path.abspath("media/loss.png"))

