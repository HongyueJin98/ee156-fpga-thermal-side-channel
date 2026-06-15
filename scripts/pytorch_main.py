import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SigmoidNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SigmoidNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.layer1(x))
        out = self.sigmoid(self.layer2(out))
        out = self.sigmoid(self.layer3(out))
        out = self.layer4(out)
        return out


def train(model, X, y, epochs=100, learning_rate=0.01):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

        losses.append(loss.item())

    return losses


def predict(model, X):
    X_tensor = torch.from_numpy(X).float()
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


# Training data
X = [[586, 586, 144, 1058, 1106, 4170], [938, 938, 144, 1058, 14767, 25229], [736, 736, 144, 1058, 12444, 18415],
     [672, 672, 144, 1058, 23382, 33178], [586, 586, 144, 1058, 1106, 4170], [1275, 1275, 144, 1058, 10226, 16648],
     [563, 563, 168, 1058, 1676, 5131], [75316, 75316, 144, 1058, 10226, 16648],
     [136, 136, 68, 33, 561, 963], [266, 66, 68, 33, 609, 1095], [170, 170, 68, 33, 910, 1148],
     [154, 154, 68, 33, 966, 1196], [146, 146, 68, 33, 1078, 1292], [454, 454, 68, 33, 610, 785],
     [18517, 18517, 74, 39, 3015, 2623], [1173, 1173, 128, 256, 11395, 10316], [1178, 1178, 128, 256, 9914, 7882],
     [1176, 1176, 128, 256, 10419, 8801], [1173, 1173, 128, 256, 11395, 10316], [1173, 1173, 128, 256, 11395, 10316],
     [1398, 1398, 128, 256, 10248, 7311], [95577, 95577, 128, 256, 27700, 19830], [1209, 17961, 0, 0, 33844, 26385],
     [897, 17649, 0, 0, 39903, 32165], [741, 17493, 0, 0, 52021, 43725], [663, 17415, 0, 0, 76257, 66845],
     [580, 17332, 0, 0, 27785, 20605], [1207, 17961, 0, 0, 33844, 26385]
     ]
'''y = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
     [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
     [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1],
     [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]
     ]'''
y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
X_train = np.array(X)
y_train = np.array(y)

# X_train = np.random.randn(28, 6)
#y_train = np.random.randint(4, size=28)
# Initialize the model
input_dim = 6
hidden_dim = 16
output_dim = 4
model = SigmoidNet(input_dim, hidden_dim, output_dim)

# Train the model
losses = train(model, X_train, y_train)

# Plot the loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test the model
X_test = X_train
y_test = y_train
y_pred = predict(model, X_test)

# Calculate accuracy for each class
accuracy = np.zeros(output_dim)
for i in range(output_dim):
    mask = y_test == i
    accuracy[i] = np.mean(y_pred[mask] == y_test[mask])
    print('Accuracy for class {}: {:.2f}%'.format(i, accuracy[i] * 100))

# Plot the accuracy
plt.bar(range(output_dim), accuracy)
plt.title('Test Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.show()
