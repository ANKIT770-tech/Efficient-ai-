#1)CIFAR-10 DATASET

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Optionally import psutil to measure memory and CPU usage.
try:
    import psutil
except ImportError:
    psutil = None


# Adjustable Parameters

adjustable_batch_size = 128
adjustable_epochs = 65
adjustable_learning_rate = 1e-3
adjustable_lambda_reg = 0.01       # Regularization weight for dynamic gating.
adjustable_weight_accuracy = 0.4   # Weight for accuracy in composite metric.
adjustable_weight_efficiency = 0.4 # Weight for FLOPs reduction in composite metric.
adjustable_weight_memory = 0.2     # Weight for memory usage penalty.
# The composite metric compares memory usage relative to the static model (reference).
# (This value will be set after evaluating the static model.)
adjustable_memory_ref = None

# 1. Define an Optimized Dynamic Block (with reduced gating branch)

class OptimizedDynamicBlock(nn.Module):
    def __init__(self, channels, dynamic=True):
        """
        A convolutional block that optionally uses a gating mechanism.
        In dynamic mode, a channel reduction (via 1x1 conv) is used before the gate.
        """
        super(OptimizedDynamicBlock, self).__init__()
        self.dynamic = dynamic
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        if self.dynamic:
            # Reduce channels to save memory in the gating branch.
            self.reduce = nn.Conv2d(channels, channels // 2, kernel_size=1)
            self.gate_fc = nn.Linear(channels // 2, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dynamic:
            # Compute the convolutional output.
            out = self.relu(self.bn(self.conv(x)))
            # Use a reduced representation for gating.
            reduced = self.reduce(x)
            gap = F.adaptive_avg_pool2d(reduced, 1).view(x.size(0), -1)
            gate = self.sigmoid(self.gate_fc(gap))
            gate_expanded = gate.view(x.size(0), 1, 1, 1)
            # Blend computed features with the identity.
            out = gate_expanded * out + (1 - gate_expanded) * x
            return out, gate
        else:
            # In static mode, always compute the full output.
            out = self.relu(self.bn(self.conv(x)))
            return out, torch.ones(x.size(0), 1, device=x.device)



# 2. Define the CNN Model Using OptimizedDynamicBlock


class OptimizedDynamicCNN(nn.Module):
    def __init__(self, num_classes=10, dynamic=True, lambda_reg=0.001):
        """
        num_classes: Number of output classes (CIFAR‑10 has 10).
        dynamic: If True, uses dynamic gating; if False, acts as a static baseline.
        lambda_reg: Regularization weight for the gate activations.
        """
        super(OptimizedDynamicCNN, self).__init__()
        self.dynamic = dynamic
        self.lambda_reg = lambda_reg

        # Initial convolution layer.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Three sequential blocks.
        self.block1 = OptimizedDynamicBlock(64, dynamic)
        self.block2 = OptimizedDynamicBlock(64, dynamic)
        self.block3 = OptimizedDynamicBlock(64, dynamic)
        # Classification head.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x, gate1 = self.block1(x)
        x, gate2 = self.block2(x)
        x, gate3 = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)
        return logits, [gate1, gate2, gate3]



# 2a. Define a LightweightCNN Class


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()

        # Initial convolution layer with fewer channels.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Reduced channels to 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Lightweight blocks with depthwise separable convolutions.
        self.block1 = self._make_depthwise_separable_block(32, 64)  # Increased channels to 64
        self.block2 = self._make_depthwise_separable_block(64, 64)
        self.block3 = self._make_depthwise_separable_block(64, 64)

        # Classification head.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_depthwise_separable_block(self, in_channels, out_channels):
        """
        Creates a depthwise separable convolution block.
        """
        return nn.Sequential(
            # Depthwise convolution.
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution.
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)
        return logits, [torch.ones(x.size(0), 1, device=x.device)] * 3  # Dummy gates for compatibility


# 3. Data Preparation (CIFAR‑10)


def get_dataloaders(batch_size=adjustable_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader



# 4. Training Function (optionally using AMP for GPU memory efficiency)


def train(model, trainloader, device, epochs=adjustable_epochs):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=adjustable_learning_rate)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs, gates = model(inputs)
                    loss = criterion(outputs, targets)
                    if isinstance(model, OptimizedDynamicCNN) and model.dynamic:
                        gate_penalty = sum(g.mean() for g in gates)
                        loss += model.lambda_reg * gate_penalty
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, gates = model(inputs)
                loss = criterion(outputs, targets)
                if isinstance(model, OptimizedDynamicCNN) and model.dynamic:
                    gate_penalty = sum(g.mean() for g in gates)
                    loss += model.lambda_reg * gate_penalty
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%")
    return model



# 5. Evaluation Function (Collects Metrics)


def evaluate(model, testloader, device):
    model.to(device)
    model.eval()
    total_time = 0.0
    total_samples = 0
    correct = 0
    gate_accum = [0.0, 0.0, 0.0]
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            start = time.time()
            outputs, gates = model(inputs)
            elapsed = time.time() - start
            total_time += elapsed

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            if isinstance(model, OptimizedDynamicCNN) and model.dynamic:
                for i, gate in enumerate(gates):
                    gate_accum[i] += gate.mean().item()
            num_batches += 1

    acc = 100. * correct / total_samples
    avg_time_ms = (total_time / total_samples) * 1000.0  # milliseconds per sample
    avg_gates = [g / num_batches for g in gate_accum] if isinstance(model, OptimizedDynamicCNN) and model.dynamic else [1.0, 1.0, 1.0]
    # Estimate theoretical FLOPs reduction from the average gate activations.
    flops_reduction = (sum(1 - a for a in avg_gates) / len(avg_gates)) * 100.0 if isinstance(model, OptimizedDynamicCNN) and model.dynamic else 0.0
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) if psutil else None
    cpu_usage = psutil.cpu_percent(interval=1) if psutil else None

    return {
        "Accuracy (%)": acc,
        "Memory (MB)": mem_usage,
        "CPU Usage (%)": cpu_usage,
        "Avg Gate Activation": avg_gates,
        "Theoretical FLOPs Reduction (%)": flops_reduction,
    }



# 6. Composite Metric Function (Including Memory Usage)


def composite_metric_with_memory(accuracy, flops_reduction, memory_usage, memory_ref,
                                 weight_accuracy=adjustable_weight_accuracy,
                                 weight_efficiency=adjustable_weight_efficiency,
                                 weight_memory=adjustable_weight_memory):
    """
    Compute a composite score that balances accuracy, efficiency, and memory usage.
    """
    norm_accuracy = accuracy / 100.0
    norm_efficiency = flops_reduction / 100.0
    norm_memory = memory_usage / memory_ref if memory_ref is not None and memory_usage is not None else 1.0
    # Subtract the memory penalty.
    return weight_accuracy * norm_accuracy + weight_efficiency * norm_efficiency - weight_memory * norm_memory



# 7. Print Metrics


def print_metrics(model_name, metrics, dynamic=True):
    print(f"\n----- {model_name} Metrics -----")
    print(f"Accuracy: {metrics['Accuracy (%)']:.2f}%")
    if metrics["Memory (MB)"] is not None:
        print(f"Memory Usage: {metrics['Memory (MB)']:.2f} MB")
    if metrics["CPU Usage (%)"] is not None:
        print(f"CPU Usage: {metrics['CPU Usage (%)']:.2f}%")
    if dynamic:
        for i, act in enumerate(metrics["Avg Gate Activation"], start=1):
            print(f"Block {i} Avg Gate Activation: {act*100:.2f}%")
        print(f"Theoretical FLOPs Reduction: {metrics['Theoretical FLOPs Reduction (%)']:.2f}%")
    # Print composite metric if a reference memory is available.
    if adjustable_memory_ref is not None:
        comp_score = composite_metric_with_memory(metrics["Accuracy (%)"],
                                                  metrics["Theoretical FLOPs Reduction (%)"],
                                                  metrics["Memory (MB)"],
                                                  adjustable_memory_ref)
        print(f"Composite Metric Score  -  Higher the Better: {comp_score:.4f}")
    print("------------------------------")



# 8. Main Function: Run Both Experiments


def main():
    global adjustable_memory_ref  # Set reference memory usage after static model evaluation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(adjustable_batch_size)

    # Experiment 1: Lightweight Model (no gating)
    print("Training Lightweight Model...")
    lightweight_model = LightweightCNN(num_classes=10)
    lightweight_model = train(lightweight_model, trainloader, device, adjustable_epochs)
    metrics_lightweight = evaluate(lightweight_model, testloader, device)
    # Set the lightweight model's memory usage as the reference.
    adjustable_memory_ref = metrics_lightweight["Memory (MB)"] if metrics_lightweight["Memory (MB)"] is not None else 1.0
    print("---------------------------------------------------------------------")
    print_metrics("Lightweight Model", metrics_lightweight, dynamic=False)
    print("---------------------------------------------------------------------")

    # Experiment 2: Dynamic Model (with optimized gating)
    print("---------------------------------------------------------------------")
    print("\nTraining Dynamic Model (with gating)...")
    print("---------------------------------------------------------------------")
    dynamic_model = OptimizedDynamicCNN(num_classes=10, dynamic=True, lambda_reg=adjustable_lambda_reg)
    dynamic_model = train(dynamic_model, trainloader, device, adjustable_epochs)
    metrics_dynamic = evaluate(dynamic_model, testloader, device)
    print_metrics("Dynamic Model", metrics_dynamic, dynamic=True)


if __name__ == "__main__":
    main()
