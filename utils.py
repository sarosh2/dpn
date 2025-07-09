import torch
import time
import numpy as np

def evaluate(model, dataloader, criterion, flatten, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).to(device) if flatten else x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_sum += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1) if outputs.dim() > 1 and outputs.shape[1] > 1 else (outputs > 0.5).float()
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 and y_batch.shape[1] > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)

        test_loss, test_acc = loss_sum / total, correct / total
    return test_loss, test_acc

def train(model, train_loader, val_loader, epochs, optimizer, criterion, flatten=True, device=None):
    train_metrics = np.empty((0, 3))  # list of (time_per_batch, loss, accuracy) per batch
    val_metrics = np.empty((0, 2))    # list of (val_loss, val_accuracy) per epoch

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        batch_metrics = np.empty((0, 3))

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).to(device) if flatten else x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            batch_start = time.time()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_start
            
            running_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1) if outputs.dim() > 1 and outputs.shape[1] > 1 else (outputs > 0.5).float()
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 and y_batch.shape[1] > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)

            new_metrics = np.array([batch_time, loss.item(), (preds == y_batch_labels).float().mean().item()]).reshape(1, -1)
            batch_metrics = np.concatenate([batch_metrics, new_metrics])

        train_metrics = np.concatenate([train_metrics, batch_metrics.sum(axis=0, keepdims=True)])

        total_time = train_metrics[-1, 0]  # Sum of all batch times
        avg_time_per_batch = total_time / batch_metrics.shape[0]
        train_loss = running_loss / total
        train_acc = correct / total

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, flatten, device)
            val_metrics = np.concatenate([val_metrics, np.array([val_loss, val_acc]).reshape(1, -1)])

        print(f"\nEpoch: {epoch} Total_Time: {total_time:.4f} Average_Time_per_batch: {avg_time_per_batch:.4f} "
            f"Train_Accuracy: {train_acc:.4f} Train_Loss: {train_loss:.4f} ", end='')
        
        if val_loader:
            print(f"Validation_Accuracy: {val_acc:.4f} Validation_Loss: {val_loss:.4f}", end='')

    return train_metrics, val_metrics

def prune_by_percent_once(percent, mask, final_weight):
    # Get the absolute values of weights where mask == 1
    masked_weights = final_weight[mask == 1].abs()

    # Sort the unmasked weights
    sorted_weights, _ = torch.sort(masked_weights)
    if sorted_weights.shape[0] != 0:
        # Determine the cutoff index for pruning
        cutoff_index = min(int(round(percent * sorted_weights.shape[0])), sorted_weights.shape[0] - 1)
        cutoff = sorted_weights[cutoff_index]
        new_mask = torch.where(final_weight.abs() <= cutoff, torch.zeros_like(mask), mask)

    else:
        new_mask = mask

    return new_mask

def prune_by_percent(model, masks, percent):

    blocks = model.layers
    for i in range(len(blocks)):
        masks[i] = prune_by_percent_once(percent, masks[i], blocks[i].weight)

    return masks

def apply_lth_pruning(model, train_loader, val_loader, epochs, optimizer, criterion, percent, rounds):
    
    original_weights = [layer.weight.clone().detach() for layer in model.layers]
    final_masks = [torch.ones_like(param) for param in original_weights]
    current_masks = [torch.ones_like(param) for param in original_weights]
    p_per_round = 1 - (1 - percent) ** (1 / rounds)

    print("Initial Validation Phase", end='')
    _, val_metrics = train(model, train_loader, val_loader, epochs, optimizer, criterion)
    val_accuracy = val_metrics[-1][1]

    for round_idx in range(rounds):
        print("\n\nPruning Round:", round_idx + 1, end='')
        current_masks = prune_by_percent(model, current_masks, p_per_round)
        pruned_weights = [w * m for w, m in zip(original_weights, current_masks)]
        
        with torch.no_grad():
            for i in range(len(model.layers)):
                model.layers[i].weight.copy_(pruned_weights[i])

        _, val_metrics = train(model, train_loader, val_loader, epochs, optimizer, criterion)
        new_val_accuracy = val_metrics[-1][1]
        
        if new_val_accuracy >= val_accuracy:
            final_masks = [m.clone() for m in current_masks]
        else:
            break

    return [w * m for w, m in zip(original_weights, final_masks)]

def merge_weights(weights, size_limit, num_classes):

    sizes = [t.shape[0] for t in weights]
    hidden_nodes = sum(sizes) - num_classes
    merged_weights = []
    start_idx = 0
    current_nodes = weights[0].shape[0]
    start_node = 0
    for i in range(1, len(weights)):
        nonzero_idx = torch.nonzero(weights[i])[-1][1].item()

        if nonzero_idx >= size_limit or (start_node < hidden_nodes and current_nodes >= hidden_nodes):
            merged_weights.append(torch.cat([t[:, :size_limit] for t in weights[start_idx:i]], dim=0))
            
            size_limit = weights[i].shape[1]
            start_idx = i
            start_node = current_nodes

        current_nodes += weights[i].shape[0]
        
    merged_weights.append(torch.cat([t[:, :size_limit] for t in weights[start_idx:]], dim=0))
    return merged_weights

def clean_weights(weights, size_limit):
    zero_indices = [i + size_limit for i, t in enumerate(weights) if torch.all(t == 0)]
    print("Empty Blocks: ", zero_indices)


    while len(zero_indices) > 0:
        weights = [t for t in weights if not torch.all(t == 0)]

        for i in range(len(weights)):
            tensor = weights[i]
            columns_to_keep = [i for i in range(tensor.shape[1]) if i not in zero_indices]
            weights[i] = tensor[:, columns_to_keep]

        zero_indices = [i + size_limit for i, t in enumerate(weights) if torch.all(t == 0)]
        print("Empty Blocks: ", zero_indices)

    return weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)