import torch, json, time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ConvolutionalBlock(nn.Module):
    
    @staticmethod
    def isOdd(number):
        return number % 2 != 0
    
    @staticmethod
    def checkConv(kernel_size, stride, padding):
        if ConvolutionalBlock.isOdd(kernel_size):
            if padding != (kernel_size - 1) // 2 or stride != 1:
                raise ValueError(
                    f"Conv: Invalid padding {padding} or stride {stride} for odd kernel size {kernel_size}."
                )
        else:
            if kernel_size != stride:
                raise ValueError(
                    f"Conv: Stride {stride} must be equal to kernel size {kernel_size} for even kernels."
                )

    @staticmethod
    def checkPool(kernel_size, stride, padding):        
        if ConvolutionalBlock.isOdd(kernel_size):
            if padding != (kernel_size - 1) // 2 or stride != 1:
                raise ValueError(
                    f"Pooling: Invalid padding {padding} or stride {stride} for odd kernel size {kernel_size}."
                )
        else:
            if kernel_size != stride:
                raise ValueError(
                    f"Pooling: Stride {stride} must be equal to kernel size {kernel_size} for even kernels."
                )
    
    
    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=3,
            kernel_size: int=3,
            stride: int=1,
            padding: int=1,
            padding_mode: str='zeros',
            activation=nn.ReLU(),
            pool_kernel_size: int=3,
            pool_stride: int=1,
            pool_padding: int=1
    ):
        super(ConvolutionalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.activation = activation
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        # Check parameters consistency
        ConvolutionalBlock.checkConv(kernel_size, stride, padding)
        ConvolutionalBlock.checkPool(pool_kernel_size, pool_stride, pool_padding)

        # Define convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding)
        )

        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        return self.conv_block(x)

    def getConvOutputShape(self, in_shape):
        return (self.out_channels,
                (in_shape[1] - self.kernel_size + 2*self.padding) // self.stride + 1,
                (in_shape[2] - self.kernel_size + 2*self.padding) // self.stride + 1)

    def getPoolOutputShape(self, in_shape):
        return (self.out_channels,
                (in_shape[1] - self.pool_kernel_size + 2*self.pool_padding) // self.pool_stride + 1,
                (in_shape[2] - self.pool_kernel_size + 2*self.pool_padding) // self.pool_stride + 1)

    def getOutputShape(self, in_shape):
        conv_shape = self.getConvOutputShape(in_shape)
        pool_shape = self.getPoolOutputShape(conv_shape)
        return pool_shape
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization for Sigmoid
                if isinstance(self.activation, nn.Sigmoid):
                    nn.init.xavier_uniform_(module.weight)
                elif isinstance(self.activation, nn.ReLU):
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm weights initialized to 1 and biases to 0
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
class CNN(nn.Module):

    def __init__(
            self,
            conv_blocks: list,
            image_shape: tuple=(1, 28, 28),
            n_classes: int=10,
            out_neurons: int=64,
            activation=nn.ReLU(),
            criterion=nn.CrossEntropyLoss(),
            dropout_rate: float=0.5,
            init_type: str='xavier',
            device: str='cpu'
    ):
        super(CNN, self).__init__()
        for conv_block in conv_blocks:
            image_shape = conv_block.getOutputShape(image_shape)
        self.out_shape = image_shape
        self.in_neurons = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.out_neurons = out_neurons
        self.n_classes = n_classes
        self.activation = activation
        self.criterion = criterion
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.device = device

        # Define layers
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fully_connected = nn.Sequential(
                nn.Linear(self.in_neurons, out_neurons),
                activation,
                nn.Dropout(dropout_rate),
                nn.Linear(out_neurons, n_classes)
        )

        # Initialize the parameters.
        self.initialize_weights(init_type)

        # Define the metrics.
        self.metrics = {
            'epochs': [], 
            'loss': {'train': [], 'val': [], 'test': None}, 
            'accuracy': {'train': [], 'val': [], 'test': None}, 
            'time': 0.0
        }

    def initialize_weights(self, init_type):
        for layer in self.fully_connected:
            if isinstance(layer, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                else:
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, a=0, b=1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(-1, self.in_neurons)
        return self.fully_connected(x)
    
    def trainBatch(self, inputs_batch, targets_batch, optimizer):
        if self.device != 'cpu':
            inputs_batch = inputs_batch.to(self.device)
            targets_batch = targets_batch.to(self.device)
        predictions_batch = self.forward(inputs_batch)          # forward pass
        loss = self.criterion(predictions_batch, targets_batch) # compute the training loss
        optimizer.zero_grad()                                   # zero the gradients
        loss.backward()                                         # backward pass
        optimizer.step()                                        # update the parameters (weights and biases)
        return loss.item()

    def predict(self, inputs):
        if self.device != 'cpu':
            inputs = inputs.to(self.device)
        self.eval()
        with torch.no_grad():
            predictions = self.forward(inputs)
        _, predictions = torch.max(predictions.cpu().data, 1)
        return predictions

    def computeLoss(self, dataloader):
        loss = 0.0
        samples = 0
        self.eval() # set the model to evaluation mode
        with torch.no_grad():   
            for inputs_batch, targets_batch in dataloader:
                batch_size = inputs_batch.size(0)
                samples += batch_size
                if self.device != 'cpu':
                    inputs_batch = inputs_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)
                predictions_batch = self.forward(inputs_batch)
                loss += self.criterion(predictions_batch, targets_batch).item() * batch_size
        return loss / samples
    
    def computeAccuracy(self, dataloader):
        correct, total = 0, 0
        self.eval() # set the model to evaluation mode
        with torch.no_grad():
            for inputs_batch, targets_batch in dataloader:
                if self.device != 'cpu':
                    inputs_batch = inputs_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)
                predictions_batch = self.forward(inputs_batch)
                correct += (predictions_batch.argmax(dim=1) == targets_batch).sum().item()
                total += inputs_batch.size(0)
        return correct / total
    
    def fit(self, train_dataloader, optimizer=optim.Adam, epochs=30, lr=1e-4, 
        regularization=0.0, val_dataloader=None, verbose=True, epoch_print=1, 
        tolerance=1e-3, patience=5):

        # Set the starting epoch
        last_epoch = self.metrics['epochs'][-1] if self.metrics['epochs'] else 0
        starting_epoch = last_epoch + 1
    
        # Set the optimizer
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)

        # Variables for early stopping
        error_loss, error_accuracy = 1, 1
        old_loss, old_accuracy = None, None
        epochs_since_improvement = 0

        # Start the training
        start_time = time.time()
        for i in range(epochs):
            self.train()
            for train_batch in train_dataloader:
                self.trainBatch(train_batch[0], train_batch[1], optimizer)

            # Evaluate the model
            self.eval()
            train_loss = self.computeLoss(train_dataloader)
            train_acc = self.computeAccuracy(train_dataloader)
            self.metrics['epochs'].append(starting_epoch + i)
            self.metrics['loss']['train'].append(train_loss)
            self.metrics['accuracy']['train'].append(train_acc)
            if val_dataloader:
                eval_loss = self.computeLoss(val_dataloader)
                eval_acc = self.computeAccuracy(val_dataloader)
                self.metrics['loss']['val'].append(eval_loss)
                self.metrics['accuracy']['val'].append(eval_acc)

                # Check early stopping conditions on eval set
                if i == 0:
                    old_loss, old_accuracy = eval_loss, eval_acc
                else:
                    error_loss = abs(eval_loss - old_loss) / old_loss
                    error_accuracy = abs(eval_acc - old_accuracy) / old_accuracy
                    old_loss, old_accuracy = eval_loss, eval_acc
            else:
                # Check early stopping conditions on train set
                if i == 0:
                    old_loss, old_accuracy = train_loss, train_acc
                else:
                    error_loss = abs(train_loss - old_loss) / old_loss
                    error_accuracy = abs(train_acc - old_accuracy) / old_accuracy
                    old_loss, old_accuracy = train_loss, train_acc
            
            if (error_loss <= tolerance) and (error_accuracy <= tolerance):
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            
            # Print the progress
            if verbose and (i + 1) % epoch_print == 0:
                eval_loss = eval_loss if val_dataloader else 'N/A'
                text = f"Epoch {starting_epoch + i}/{starting_epoch + epochs - 1}: "
                text += f"Loss ({train_loss:.4g}, {eval_loss:.4g}) \t "
                text += f"Accuracy ({100*train_acc:.2f}%, {100*eval_acc:.2f}%)"
                print(text)

            # Early stopping check
            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered after {i + 1} epochs.")
                break

        self.metrics['time'] += time.time() - start_time

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, model_path: str="cnn.pth", metrics_path: str="metrics.txt"):
        torch.save(self.state_dict(), model_path)
        with open(metrics_path, 'w') as f:
            f.truncate()
            json.dump(self.metrics, f)
        f.close()

    def load(self, model_path: str="cnn.pth", metrics_path: str="metrics.txt"):
        self.load_state_dict(torch.load(model_path))
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
        f.close()
        self.to(self.device)

    def plotMetrics(self, file_path: str="figures/metrics.png", figsize: tuple=(10, 4), save: bool=False):
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # Plot loss
        axs[0].plot(self.metrics['epochs'], self.metrics['loss']['train'], label=f"Training")
        axs[0].plot(self.metrics['epochs'], self.metrics['loss']['val'], label=f"Validation")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].legend(loc='best')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot accuracy
        axs[1].plot(self.metrics['epochs'], self.metrics['accuracy']['train'], label=f"Training")
        axs[1].plot(self.metrics['epochs'], self.metrics['accuracy']['val'], label=f"Validation")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend(loc='best')
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        
        if save:
            plt.savefig(file_path, dpi=100, facecolor='w', edgecolor='w')
            plt.close()
        else:
            plt.show()