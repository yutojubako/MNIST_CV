from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Datasetの作成
class MNISTDataset(Dataset):
    def __init__(self, is_test=False, transform=None):
        mnist = fetch_openml("mnist_784", cache=False)
        # change the data type from DataFrame to numpy
        all_images = mnist.data.astype("float32").to_numpy()
        all_labels = mnist.target.astype("int64").to_numpy()

        # データは70000存在．60000を学習用，10000をテスト用に割り当てる
        images_train, images_test, labels_train, labels_test = train_test_split(
            all_images, all_labels, test_size=10000, random_state=42
        )
        if is_test:
            self.images = images_test
            self.labels = labels_test
        else:
            self.images = images_train
            self.labels = labels_train

        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}
    
class ImageTransfrom:
    def __init__(self):
        self.transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __call__(self, image):
        return self.transfrom(image)
    
# Compare this snippet from src/model.py: