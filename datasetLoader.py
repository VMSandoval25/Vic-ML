from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

class DatasetLoader:
    def __init__(self, root_dir, batch_size=512, img_size=(256, 256), val_split=0.2):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        full_train_dataset = ImageFolder(root=f'{self.root_dir}/train', transform=self.transform)   
        train_size = int((1 - self.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        test_dataset = ImageFolder(root=f'{self.root_dir}/test', transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
