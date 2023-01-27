import os
import sys
import torch
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pytz
from datetime import datetime
from model import CNNModel
from dataset import MNISTDataset, ImageTransfrom
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix as cm



class MNIST():
    def __init__(self, args, BATCH_SIZE, EPOCH_NUM, SEED) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = CNNModel(1, 10).to(self.device)
        self.args = args
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_NUM = EPOCH_NUM
        self.SEED = SEED
    def train_model(self):
        # to log the best loss
        best_loss = sys.maxsize
        history = {"train_loss": [], "val_loss": []}
        
        # Prepare the model
        self.cnn_model.to(self.device)
        
        train_val_dataset = MNISTDataset(is_test=False, transform=ImageTransfrom())
        n_samples = len(train_val_dataset) # 60000
        train_size = int(n_samples * 0.8) # 48000
        val_size = n_samples - train_size # 12000
        
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
        
        # optimizer
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=0.001)
        
        # fix the random seed
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
        valid_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # TODO:wandb init
        jst = pytz.timezone('Asia/Tokyo')
        now = datetime.now(jst)
        wandb.init(project="CV_MNIST", entity="keiomobile2", name=now.strftime("%Y-%m-%d %H:%M:%S"), config=self.args)
        
        for epoch in range(1, self.EPOCH_NUM + 1):
            train_loss = 0.0
            val_loss = 0.0
            
            # train
            self.cnn_model.train()
            
            for batch in train_dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self.cnn_model(images)
                
                # if you wanna other loss function, you should change below.
                # LOSS FUNCTION
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss = train_loss / len(train_dataloader)
            history["train_loss"].append(train_loss)
            
            #validation
            with torch.no_grad():
                self.cnn_model.eval()
                
                for batch in valid_dataloader:
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = self.cnn_model(images)
                    
                    # if you wanna other loss function, you should change below.
                    # LOSS FUNCTION
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # wandb log of validation loss
                val_loss = val_loss / len(valid_dataloader)
                history["val_loss"].append(val_loss)
                # save the best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.cnn_model.state_dict(), "best_model.pth")
                    torch.save(self.cnn_model.to('cpu').state_dict(), "best_model_cpu.pth")
                    self.cnn_model.to(self.device)
                    print(
                        f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                        )
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        self.plot_loss(history, save_dir="./results/loss.png")
        
    def load_model(self, filepath):
        # load the best model
        self.cnn_model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.cnn_model.to(self.device)
        return self.cnn_model
    
    
    def test_model(self):
        # Make test dataset
        test_dataset = MNISTDataset(is_test=True, transform=ImageTransfrom())
        
        # Make test dataloader
        # if you compare with NN model, you should change batch size.
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1, drop_last=False)
        
        start = time.time()
        self.cnn_model = self.load_model("./best_model.pth")
        
        label_list = []
        answer_list = []
        
        with torch.no_grad():
            #change mode to evaluation
            self.cnn_model.eval()
            
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.cnn_model(images)
                
                np_outputs = torch.squeeze(outputs).cpu().numpy().copy()
                np_labels = torch.squeeze(labels).cpu().numpy().copy()
                label_list.append(np_labels)
                answer_list.append(np.argmax(np_outputs, axis=1))
                
        label_list = np.concatenate(label_list, axis=0)
        answer_list = np.concatenate(answer_list, axis=0)
        
        elapsed_time = time.time() - start
        print(f"elapsed_time: {elapsed_time:.4f} [sec]")
        
        accuracy = self.create_confusion_matrix(
            label_list, answer_list, save_dir="./results/confusion_matrix.png"
            )
        print(f"accuracy: {accuracy:.4f}")
        
        data = {f"elapsed_time": elapsed_time, f"accuracy": accuracy}
        self.write_data_as_json(file_dir="./results/result.json", key="CNN", data=data)
        
    def plot_loss(self, history, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'loss.png')
        epoch_list = list(range(1, self.EPOCH_NUM + 1))
        fig = plt.figure().add_subplot(111)
        fig.plot(epoch_list, history["train_loss"], label="train_loss")
        fig.plot(epoch_list, history["val_loss"], label="val_loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def create_confusion_matrix(self, y_true, y_pred, save_dir):
        confmat = cm(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        total = np.sum(confmat)
        correct = 0
        
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")
                if i == j:
                    correct += confmat[i, j]
        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 10, 1))
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.savefig(save_dir)
        plt.show()
        plt.close()
        acc = correct / total * 100
        return acc
    
    def write_data_as_json(self, file_dir, key, data):
        if os.path.exists(file_dir):
            with open(file_dir, "r") as f:
                json_data = json.load(f)
        else:
            json_data = {}
        json_data[key] = data
        with open(file_dir, "w") as f:
            json.dump(json_data, f, indent=4)