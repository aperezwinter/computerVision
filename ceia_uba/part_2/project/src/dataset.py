import torch, random, os, cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2


class CardsDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
class Cards(object):
    BLACK_PIXEL = 0
    WHITE_PIXEL = 255

    def __init__(
            self,
            transform: transforms.Compose,
            train_path: str="dataset/train/", 
            valid_path: str="dataset/valid/", 
            test_path: str="dataset/test/", 
            device: str="cpu",
    ) -> None:
        self.transform = transform
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.train_data = datasets.ImageFolder(root=train_path, transform=self.transform)
        self.valid_data = datasets.ImageFolder(root=valid_path, transform=self.transform)
        self.test_data = datasets.ImageFolder(root=test_path, transform=self.transform)
        self.device = device

        self.n_classes = len(self.train_data.classes)
        self.classes = self.train_data.classes
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)

        self.img_channels = self.train_data[0][0].shape[0]
        self.img_width = self.train_data[0][0].shape[1]
        self.img_height = self.train_data[0][0].shape[2]
        
    def __len__(self):
        return self.n_train, self.n_valid, self.n_test
    
    def __getitem__(self, idx):
        return self.train_data[idx]
    
    def getDatasets(self):
        # Training dataset
        inputs, labels = [], []
        for img, label in self.train_data:
            inputs.append(img.type(torch.float32))
            labels.append(torch.tensor(label, dtype=torch.long))
        self.train_dataset = CardsDataset(inputs, labels)

        # Validation dataset
        inputs, labels = [], []
        for img, label in self.valid_data:
            inputs.append(img.type(torch.float32))
            labels.append(torch.tensor(label, dtype=torch.long))
        self.valid_dataset = CardsDataset(inputs, labels)

        # Test dataset
        inputs, labels = [], []
        for img, label in self.test_data:
            inputs.append(img.type(torch.float32))
            labels.append(torch.tensor(label, dtype=torch.long))
        self.test_dataset = CardsDataset(inputs, labels)
    
    def getDataloaders(self, batch_size: list=[128, 32, 32]):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size[0], shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size[1], shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size[2], shuffle=False)

    def getDataByClass(self, classes: list, dataset: str="train"):
        if dataset == "train":
            return [sample for sample in self.train_data if sample[1] in classes]
        elif dataset == "valid":
            return [sample for sample in self.valid_data if sample[1] in classes]
        elif dataset == "test":
            return [sample for sample in self.test_data if sample[1] in classes]
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
    def getLabelsByClass(self, classes: list, dataset: str="train"):
        if dataset == "train":
            return [sample[1] for sample in self.train_data if sample[1] in classes]
        elif dataset == "valid":
            return [sample[1] for sample in self.valid_data if sample[1] in classes]
        elif dataset == "test":
            return [sample[1] for sample in self.test_data if sample[1] in classes]
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
    def getImagesByClass(self, classes: list, dataset: str="train"):
        if dataset == "train":
            return [sample[0] for sample in self.train_data if sample[1] in classes]
        elif dataset == "valid":
            return [sample[0] for sample in self.valid_data if sample[1] in classes]
        elif dataset == "test":
            return [sample[0] for sample in self.test_data if sample[1] in classes]
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
    def getAugmentedData(self, image_height, image_width):

        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=0.8, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0
            ),  # Shift, scale and rotate
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),  # Brightness and contrast
            A.HorizontalFlip(p=0.5),  # Horizontal flip
            A.Resize(image_height, image_width),  # Resize
            ToTensorV2()  # Convert to tensor
        ])
    
    def augmentDataset(self, loops: int=1, join: bool=False):
        augmented_train_data = []
        augmented_valid_data = []
        augmented_test_data = []
        for sample in self.train_data:
            image = np.transpose(sample[0].numpy(), (1, 2, 0))
            width, height = image.shape[1], image.shape[0]
            for _ in range(loops):
                image_augmented = self.getAugmentedData(height, width)(image=image)['image']
                augmented_train_data.append((image_augmented, sample[1]))
        for sample in self.valid_data:
            image = np.transpose(sample[0].numpy(), (1, 2, 0))
            width, height = image.shape[1], image.shape[0]
            for _ in range(loops):
                image_augmented = self.getAugmentedData(height, width)(image=image)['image']
                augmented_valid_data.append((image_augmented, sample[1]))
        for sample in self.test_data:
            image = np.transpose(sample[0].numpy(), (1, 2, 0))
            width, height = image.shape[1], image.shape[0]
            for _ in range(loops):
                image_augmented = self.getAugmentedData(height, width)(image=image)['image']
                augmented_test_data.append((image_augmented, sample[1]))
        
        if join:
            # Join the augmented data with the original data and rewrite the datasets
            self.train_data = augmented_train_data + list(self.train_data)
            self.valid_data = augmented_valid_data + list(self.valid_data)
            self.test_data = augmented_test_data + list(self.test_data)
            self.n_train = len(self.train_data)
            self.n_valid = len(self.valid_data)
            self.n_test = len(self.test_data)
        else:
            return augmented_train_data, augmented_valid_data, augmented_test_data
        
    def toGrayScale(self, image):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        return image
    
    def turnDataset2GrayScale(self):
        train_data, valid_data, test_data = [], [], []

        for (image, label) in self.train_data:
            image_bw = self.toGrayScale(image)
            train_data.append((image_bw, label))

        for (image, label) in self.valid_data:
            image_bw = self.toGrayScale(image)
            valid_data.append((image_bw, label))

        for (image, label) in self.test_data:
            image_bw = self.toGrayScale(image)
            test_data.append((image_bw, label))

        # Update data
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        # Refresh image variables
        self.img_channels = self.train_data[0][0].shape[0]
        self.img_width = self.train_data[0][0].shape[1]
        self.img_height = self.train_data[0][0].shape[2]
    
    def plotAverageImages(self, file_path: str="figures/average_images.png"):
        # Get the images by class
        images_by_class = {i: [] for i in range(self.n_classes)}
        for sample in self.train_data:
            images_by_class[sample[1]].append(sample[0])
        for card_class, images in images_by_class.items():
            images_by_class[card_class] = torch.stack(images, dim=0)
        # Get the average images
        average_images = {i: images_by_class[i].mean(dim=0) for i in range(self.n_classes)}
        # Plot the average images
        n_rows, n_cols = 7, 8
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                img_index = i*n_cols + j
                if img_index < self.n_classes:
                    axs[i, j].imshow(average_images[img_index].permute(1, 2, 0).numpy())
                    axs[i, j].axis('off')
                    axs[i, j].set_title(self.classes[img_index], fontsize=8)
                else:
                    axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, facecolor='white', edgecolor='white')
        plt.close()

    def plotHeatMaps(self, file_path: str="figures/heat_maps.png"):
        # Get the heat maps by class
        heat_maps = {i: [] for i in range(self.n_classes)}
        for sample in self.train_data:
            heat_maps[sample[1]].append(sample[0])
        for card_class, images in heat_maps.items():
            heat_maps[card_class] = torch.stack(images, dim=0)
            heat_maps[card_class] = heat_maps[card_class].mean(dim=0)
            heat_maps[card_class] = heat_maps[card_class].mean(dim=0)
        # Plot the heat maps
        n_rows, n_cols = 7, 8
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                img_index = i*n_cols + j
                if img_index < self.n_classes:
                    img = heat_maps[img_index].cpu().numpy()
                    axs[i, j].imshow(img, cmap='hot')
                    axs[i, j].axis('off')
                    axs[i, j].set_title(self.classes[img_index], fontsize=8)
                else:
                    axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, facecolor='white', edgecolor='white')
        plt.close()
    
    def plotClassesHistogram(self, n_classes: int=-1, file_path: str="figures/class_histogram.png", percentage: bool=False):
        n_classes = n_classes if n_classes != -1 else self.n_classes
        
        # Get class counts for the datasets
        train_labels = [sample[1] for sample in self.train_data if sample[1] in range(n_classes)]
        valid_labels = [sample[1] for sample in self.valid_data if sample[1] in range(n_classes)]
        test_labels = [sample[1] for sample in self.test_data if sample[1] in range(n_classes)]

        train_counts = np.unique(train_labels, return_counts=True)[1]
        valid_counts = np.unique(valid_labels, return_counts=True)[1]
        test_counts = np.unique(test_labels, return_counts=True)[1]

        if percentage:
            train_counts = 100 * train_counts / self.n_train
            valid_counts = 100 * valid_counts / self.n_valid
            test_counts = 100 * test_counts / self.n_test
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_classes)
        width = 0.25
        ylabel = 'Percentage (%)' if percentage else 'Counts'
        title = f'Class Distribution - {n_classes} classes'

        ax.bar(x - width, train_counts, width, label='Train')
        ax.bar(x, valid_counts, width, label='Validation')
        ax.bar(x + width, test_counts, width, label='Test')

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes[:n_classes], rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.legend()

        plt.tight_layout()
        plt.savefig(file_path, dpi=300, facecolor='white', edgecolor='white')
        plt.close()

    def plotRandomSamples(self, n_samples: int=5, n_classes: int=-1, file_path: str="figures/random_samples.png"):
        random_files = {}
        n_classes = n_classes if n_classes != -1 else self.n_classes
        random_classes = random.sample(self.classes, n_classes)

        for class_name in random_classes:
            class_path = os.path.join(self.train_path, class_name)
            files_by_class = os.listdir(class_path)
            sampled_files = random.sample(files_by_class, n_samples)
            random_files[class_name] = [os.path.join(class_path, f) for f in sampled_files]

        # Plot each row a different class
        fig, axs = plt.subplots(n_classes, n_samples, figsize=(n_samples, n_classes))
        for i in range(n_classes):
            for j in range(n_samples):
                image_path = random_files[random_classes[i]][j]
                axs[i, j].imshow(mpimg.imread(image_path))
                axs[i, j].axis('off')
        
        plt.suptitle(f"Random samples: {n_samples} images - {n_classes} classes")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, facecolor='white', edgecolor='white')
        plt.close()

    def plotStatsDistribution(self, file_path: str="figures/stats_distribution.png", n_bins: int=20):
        # Get the image's stats by class (mean and std)
        images_by_class = {i: [] for i in range(self.n_classes)}
        stats_by_class = {i: None for i in range(self.n_classes)}

        for sample in self.train_data:
            images_by_class[sample[1]].append(sample[0])

        for card_class, images in images_by_class.items():
            images_by_class[card_class] = torch.stack(images, dim=0)
            stats_by_class[card_class] = (Cards.WHITE_PIXEL*images_by_class[card_class].mean(dim=(0, 2, 3)), 
                                          Cards.WHITE_PIXEL*images_by_class[card_class].std(dim=(0, 2, 3)))
        for card_class, stats in stats_by_class.items():
            stats_by_class[card_class] = (stats[0].type(torch.long), stats[1].type(torch.long))

        # Create lists to store the values
        means = {"R": [], "G": [], "B": []}
        stds = {"R": [], "G": [], "B": []}

        for card_class, stats in stats_by_class.items():
            means["R"].append(stats[0][0].item())
            means["G"].append(stats[0][1].item())
            means["B"].append(stats[0][2].item())
            stds["R"].append(stats[1][0].item())
            stds["G"].append(stats[1][1].item())
            stds["B"].append(stats[1][2].item())
        
        # Plot a normalized distribution of the means and stds
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs[0, 0].hist(means["R"], bins=n_bins, density=True, edgecolor='red', color='red', alpha=0.7)
        axs[0, 0].set_title('$\mu_R$')
        axs[0, 1].hist(means["G"], bins=n_bins, density=True, edgecolor='green', color='green', alpha=0.7)
        axs[0, 1].set_title('$\mu_G$')
        axs[0, 2].hist(means["B"], bins=n_bins, density=True, edgecolor='blue', color='blue', alpha=0.7)
        axs[0, 2].set_title('$\mu_B$')
        axs[1, 0].hist(stds["R"], bins=n_bins, density=True, edgecolor='red', color='red', alpha=0.7)
        axs[1, 0].set_title('$\sigma_R$')
        axs[1, 1].hist(stds["G"], bins=n_bins, density=True, edgecolor='green', color='green', alpha=0.7)
        axs[1, 1].set_title('$\sigma_G$')
        axs[1, 2].hist(stds["B"], bins=n_bins, density=True, edgecolor='blue', color='blue', alpha=0.7)
        axs[1, 2].set_title('$\sigma_B$')

        for ax in axs.flatten():
            ax.set_xlabel('Pixel intensity')
            ax.set_ylabel('Density')

        plt.tight_layout()
        plt.savefig(file_path, dpi=300, facecolor='white', edgecolor='white')
        plt.close()