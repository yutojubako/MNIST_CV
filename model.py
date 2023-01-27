from torch import nn

class CNNModel(nn.Module):
    def __init__(self, input_shape=1, num_classes=10):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(input_shape, num_classes , kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.dropout2 = nn.Dropout2d(p=0.5) # p is default
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        
        # Block 3
        self.fc3 = nn.Linear(20 * 4 * 4, 50)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(p=0.5)
        
        # Block 4
        # because num_classes = 10
        self.fc4 = nn.Linear(50, num_classes)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Block 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.fc4(x)
        # x = self.softmax(x)
        
        return x