import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

test_dir = r'C:\Users\jithi\OneDrive\Desktop\ML project\microsoft-malware-prediction\test.csv'
def get_csv_submission(predictions, MachineIdentifier, filename="submission.csv"):
    submission = pd.DataFrame({
        'MachineIdentifier': MachineIdentifier,
        'HasDetections': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file '{filename}' created successfully.")


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

import torch
from tqdm import tqdm

def train_and_evaluate_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    epochs,
    early_stopping_patience=7
):
    # print(f"Training on {torch.cuda.get_device_name(0)}")
    model = model.to('cuda')
    best_test_loss = float('inf')
    best_test_accuracy = 0
    best_model_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    lr_dropped = False  # Flag to prevent multiple resets after a single LR drop

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    lrs = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}') as pbar:
            lrs.append(current_lr)
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                batch_correct = (predictions == y_batch).sum().item()
                batch_total = y_batch.numel()
                batch_acc = (batch_correct / batch_total) * 100

                train_correct += batch_correct
                train_total += batch_total
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        train_loss.append(avg_loss)
        train_accuracy.append(train_acc)

        # Evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0
        total_test_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                batch_correct = (predictions == y_batch).sum().item()
                batch_total = y_batch.numel()
                test_correct += batch_correct
                test_total += batch_total

        test_loss_value = total_test_loss / len(test_loader)
        test_acc = test_correct / test_total
        print(f"Test Loss: {test_loss_value:.4f}, Test Accuracy: {test_acc*100:.2f}%")
        test_loss.append(test_loss_value)
        test_accuracy.append(test_acc)

        # Save best model
        if test_loss_value < best_test_loss:
            best_test_loss = test_loss_value
            best_model_weights = model.state_dict().copy()

        scheduler.step(test_loss_value)

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"Learning rate changed from {current_lr:.6f} to {new_lr:.6f}")
            if not lr_dropped:
                epochs_without_improvement = 0  # Reset patience after LR drop
                lr_dropped = True  # Prevent repeated resets for same drop

        # Early stopping tracking
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            lr_dropped = False  # Allow future reset if another LR drop happens
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best model was from epoch {best_epoch}.")
            break

    return best_model_weights, train_loss, test_loss, train_accuracy, test_accuracy, lrs

