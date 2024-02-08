from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

class DatasetLoader:
    def __init__(self, root_dir, batch_size=512, img_size=(256, 256), val_split=0.2, num_workers = 2):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        full_train_dataset = ImageFolder(root=f'{self.root_dir}/train', transform=self.transform)   
        train_size = int((1 - self.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # subset , _ = random_split(full_train_dataset, [0.75, .25])
        
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        # total_length = len(full_train_dataset)
        # subset_length = int(0.45 * total_length)  # 75% of the total length
        # remaining_length = total_length - subset_length  # Remaining 25%
        # subset, _ = random_split(full_train_dataset, [subset_length, remaining_length])
        # train_size = int((1 - self.val_split) * subset_length)
        # val_size = subset_length - train_size
        # train_dataset, val_dataset = random_split(subset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        test_dataset = ImageFolder(root=f'{self.root_dir}/test', transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
