import torch
import time

def evaluate(model, dataloader, criterion, flatten=True):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).cuda() if flatten else x_batch.cuda(), y_batch.cuda()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_sum += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)
    return loss_sum / total, correct / total

def train(model, train_loader, val_loader, test_loader, epochs, optimizer, criterion, flatten=True):
    train_metrics = []  # list of (time_per_batch, loss, accuracy) per batch
    val_metrics = []    # list of (val_loss, val_accuracy) per epoch
    test_metrics = []   # list of (test_loss, test_accuracy)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        batch_times = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).cuda() if flatten else x_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()

            batch_start = time.time()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_start
            
            running_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)

            batch_times.append(batch_time)
            train_metrics.append((batch_time, loss.item(), (preds == y_batch_labels).float().mean().item()))

        total_time = sum(batch_times)
        avg_time_per_batch = total_time / len(batch_times)
        train_loss = running_loss / total
        train_acc = correct / total

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            val_metrics.append((val_loss, val_acc))

        print("")
        print(f"Epoch: {epoch} Total_Time: {total_time:.4f} Average_Time_per_batch: {avg_time_per_batch:.4f} "
            f"Train_Accuracy: {train_acc:.4f} Train_Loss: {train_loss:.4f} ", end='')
        
        if val_loader:
            print(f"Validation_Accuracy: {val_acc:.4f} Validation_Loss: {val_loss:.4f}", end='')
        
    # Finally test
    test_loss, test_acc = evaluate(model, test_loader, criterion, flatten)
    test_metrics.append((test_loss, test_acc))

    print("")
    print(f"Test_Accuracy:  {test_acc:.4f} Test_Loss:  {test_loss}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

    return train_metrics, val_metrics, test_metrics