import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import traceback

class Trainer:
    def __init__(self, model, datasets, save_path, lr=0.001, decay_rate=1e-5):
        self.model = model
        self.datasets = datasets
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay_rate)
        self.save_path = save_path
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = -1

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            total_batches = len(self.datasets['train'])
            start_time = time.time()
            
            progress_bar = tqdm(enumerate(self.datasets['train']), total=total_batches, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, (images, labels) in progress_bar:
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    # print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    
                    progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}')
                except Exception as e:
                    print(f'Exception occured during epoch {epoch+1}: {e}')
                    print(traceback.format_exc())
            try:
                val_loss, val_accuracy = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                avg_loss = running_loss / total_batches
                end_time = time.time()
                epoch_time = end_time - start_time

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f} sec')
                self.save_model(val_accuracy, epoch, avg_loss)
                self.plot_metrics()
            except Exception as e:
                print(f'Exception after epoch: {e}')
                print(traceback.format_exc())
    
    def save_model(self, current_accuracy, epoch, avg_loss):
        best_model_path = os.path.join(self.save_path,f'art_model_best.pth')
        model_path = os.path.join(self.save_path,f'art_model_e_{epoch+1}')
        if current_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
        if epoch + 1 % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
        

    def validate(self):
        self.model.eval()
        total = 0
        correct = 0
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in self.datasets['val']:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = validation_loss / len(self.datasets['val'])
        accuracy = 100 * correct / total
        return avg_val_loss, accuracy
    
    def plot_metrics(self):
        # Plot validation loss
        save_path = '/Users/victorsandoval/personal/code/val_plot.png'
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()
        
