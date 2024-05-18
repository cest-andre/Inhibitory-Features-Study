import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #   baseline
            nn.MaxPool2d(kernel_size=3, stride=2),
            #   more pool
            # nn.MaxPool2d(kernel_size=6, stride=4),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #   baseline
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #   big_kernel
            # nn.Conv2d(256, 256, kernel_size=7, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        #   Global avg pool.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    #   TODO: try to distribute training across two GPUs.
    batch_size = 256
    IMAGE_SIZE = 224
    epochs = 10
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    # create a dataset object for the ImageNet dataset
    imnet_folder = r"/media/andrelongon/DATA/imagenet/train"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    #   Shuffle due to taking subset.
    # torch.manual_seed(0)
    trainloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(imagenet_data), drop_last=False)

    model = AlexNet().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    #   For loading:   https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial
    #   For training:  https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # ddp_model.load_state_dict(
    #     torch.load(CHECKPOINT_PATH, map_location=map_location))

    # states = torch.load("/media/andrelongon/DATA/imnet_weights/imnet_alexnet_big_kernel_4ep.pth")
    # model.load_state_dict(states)

    criterion = nn.CrossEntropyLoss().to(device_id)
    optimizer = optim.RAdam(ddp_model.parameters())

    for ep in range(0, epochs):
        running_loss = 0.0
        trainloader.sampler.set_epoch(ep)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device_id), labels.to(device_id)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')

        # print('Finished Training')
        #   NOTE:  less pool with have last two removed.
        #          Also try a more maxpooled model (maybe 4x pool before last conv)
        if rank == 0:
            torch.save(ddp_model.state_dict(), f"/media/andrelongon/DATA/imnet_weights/imnet_gap_alexnet_{1+ep}ep.pth")

        imnet_val_folder = r"/home/andrelongon/Documents/data/imagenet/val"
        imnet_val_data = torchvision.datasets.ImageFolder(imnet_val_folder, transform=transform)
        # torch.manual_seed(0)
        val_loader = torch.utils.data.DataLoader(imnet_val_data, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(imnet_val_data), drop_last=False)

        top1_correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device_id), labels.to(device_id)
                images = norm_transform(images)

                outputs = ddp_model(images)
                _, top1_predicted = torch.topk(outputs, k=1, dim=1)
                _, top5_predicted = torch.topk(outputs, k=5, dim=1)

                top1_correct += (top1_predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
                top5_correct += (top5_predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        print(f'Imnet val top 1: {100 * top1_correct // total} %')
        print(f'Imnet val top 5: {100 * top5_correct // total} %')

    dist.destroy_process_group()