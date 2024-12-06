import time, json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):

    def __init__(self, in_features: int=3, activation=nn.ReLU(), device: str='cpu'):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_features, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            activation,
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            activation,
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            activation,
            nn.ConvTranspose2d(16, in_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # output range [0, 1]
        )

        self.criterion = nn.MSELoss()
        self.device = device
        self.metrics = {
            'epochs': [], 
            'loss': {'train': [], 'val': [], 'test': None}, 
            'time': 0.0
        }

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def trainBatch(self, images_batch, optimizer):
        if self.device != 'cpu':
            images_batch = images_batch.to(self.device)
        _, decoded_batch = self.forward(images_batch)
        optimizer.zero_grad()
        loss = self.criterion(decoded_batch, images_batch)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def computeLoss(self, dataloader):
        loss = 0.0
        samples = 0
        self.eval() # set the model to evaluation mode
        with torch.no_grad():   
            for images_batch, _ in dataloader:
                if self.device != 'cpu':
                    images_batch = images_batch.to(self.device, non_blocking=True)
                _, decoded_batch = self.forward(images_batch)                           # forward pass
                batch_loss = self.criterion(decoded_batch, images_batch).item()         # compute the loss
                loss += batch_loss * images_batch.size(0)                               # accumulate the weighted loss
                samples += images_batch.size(0)                                         # accumulate the number of samples
        return loss / samples
    
    def fit(self, train_dataloader, optimizer=optim.Adam, epochs=30, lr=1e-4, 
        regularization=0.0, eval_dataloader=None, verbose=True, epch_print=1, 
        tolerance=1e-3, patience=5):

        # Set the starting epoch
        last_epoch = self.metrics['epochs'][-1] if self.metrics['epochs'] else 0
        starting_epoch = last_epoch + 1
    
        # Set the optimizer
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)

        # Variables for early stopping
        error_loss = 1
        old_loss = None
        epochs_since_improvement = 0

        # Start the training
        start_time = time.time()
        for i in range(epochs):
            self.train()
            for images_batch, _ in train_dataloader:
                self.trainBatch(images_batch, optimizer)
                if self.device == "gpu":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                else:
                    pass

            # Evaluate the model
            self.eval()
            train_loss = self.computeLoss(train_dataloader)
            self.metrics['epochs'].append(starting_epoch + i)
            self.metrics['loss']['train'].append(train_loss)
            if eval_dataloader:
                eval_loss = self.computeLoss(eval_dataloader)
                self.metrics['loss']['val'].append(eval_loss)

                # Check early stopping conditions on eval set
                if i == 0:
                    old_loss = eval_loss
                else:
                    error_loss = abs(eval_loss - old_loss) / old_loss
                    old_loss = eval_loss
            else:
                # Check early stopping conditions on train set
                if i == 0:
                    old_loss = train_loss
                else:
                    error_loss = abs(train_loss - old_loss) / old_loss
                    old_loss = train_loss
            
            if error_loss <= tolerance:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            
            # Print the progress
            if verbose and (i + 1) % epch_print == 0:
                eval_loss = eval_loss if eval_dataloader else 'N/A'
                text = f"Epoch {starting_epoch + i}/{starting_epoch + epochs}: "
                text += f"Loss ({train_loss:.4g}, {eval_loss:.4g})"
                print(text)

            # Early stopping check
            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered after {i + 1} epochs.")
                break

        self.metrics['time'] += time.time() - start_time

    def save(self, model_path: str="models/autoencoder.pth", metrics_path: str="metrics/autoencoder.txt"):
        torch.save(self.state_dict(), model_path)
        with open(metrics_path, 'w') as f:
            f.truncate()
            json.dump(self.metrics, f)
        f.close()

    def load(self, model_path: str="models/autoencoder.pth", metrics_path: str="metrics/autoencoder.txt"):
        self.load_state_dict(torch.load(model_path))
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
        f.close()
        self.to(self.device)

    def showEncodedImage(self, image: torch.Tensor):
        if self.device != 'cpu':
            image = image.to(self.device)
        _, decoded = self.forward(image)

        image = image[0].permute(1, 2, 0).detach().cpu().numpy()
        decoded = decoded[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Plot in 2 subplots horizontally
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[0].axis('off')
        axs[1].imshow(decoded)
        axs[1].set_title("Reconstructed")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()