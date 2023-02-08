class BasicExample:
    def __init__(
        self,
        setting1,
        setting2,
    ):
        self.setting1 = setting1
        self.setting2 = setting2

    def run(self):
        print("Running example")
        print(f"{self.setting1=}")
        print(f"{self.setting2=}")


class MNISTExample:
    def __init__(self, device='cpu', batch_size=64, num_epochs=10):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def run(self): 
        print("Running MNIST example")
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        from torch.utils.data import DataLoader
        from torch import nn
        from torch import optim
        import torch 
        from torch.nn import functional as F
        from tqdm import tqdm
        import wandb

        wandb.init(project="mnist-example")

        train_dataset = MNIST(
            root="data", train=True, transform=ToTensor(), download=True
        )
        test_dataset = MNIST(
            root="data", train=False, transform=ToTensor(), download=True
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
                return output

        model = Net().to(self.device)
        wandb.watch(model, log="all")

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        def train(model, device, train_loader, optimizer, epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                wandb.log({"loss": loss})

        def test(model, device, test_loader):
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc='Testing'):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    pred = output.argmax(
                        dim=1, keepdim=True
                    )  # get the index of the max log-probability

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            wandb.log({"test_loss": test_loss, "test_accuracy": correct / len(test_loader.dataset)})


        for epoch in range(1, self.num_epochs + 1):
            train(model, self.device, train_loader, optimizer, epoch)
            test(model, self.device, test_loader)

    