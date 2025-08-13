import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import warnings

import data_processing


warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16
lr = 0.005
scalar = MinMaxScaler()

features = data_processing.get_features()
labels = data_processing.get_product_data()
feature_corr = features.corr(method="pearson")
data_processing.show_derived_data(150) # Tilt
data_processing.show_derived_data(151) # offset

print(features.head())
plt.figure(figsize=(8,6))
corr_mat = feature_corr.where(np.tril(np.ones(feature_corr.shape), k=0).astype(bool))
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("pearson correlation among a few possible features")
plt.show()

X = np.array(features, dtype=np.float32)
X = scalar.fit_transform(X)
Y = np.array(labels["error_type"], dtype=np.float32).reshape(-1,1)
X_train, X_test, Y_train, Y_test = data_processing.split_data(X, Y)
X_shuffle = X_train.copy()
acc_lst = []
X_shuffle = torch.tensor(X_shuffle, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=False)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32, requires_grad=False)
train_data, test_data = TensorDataset(X_shuffle, Y_train), TensorDataset(X_test, Y_test)
class Classification_Model(nn.Module):
    def __init__(self,
                 input_dim=6,
                 output_dim=4
                 ):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(16, output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.out(x)
        return x

model = Classification_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
num_epochs = 300
train_size = X_train.shape[0]/batch_size
for epoch in tqdm(range(num_epochs)):
    batch_loss = 0
    batch_acc = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.train()
        y_batch = y_batch.squeeze().long()

        out = model(x_batch)
        loss = criterion(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss
        batch_acc += accuracy_score(y_batch.cpu(), np.argmax(out.cpu().detach().numpy(), axis=1))

    epoch_loss = batch_loss/train_size
    train_loss.append(epoch_loss.detach().item())
    train_accuracy.append(batch_acc/train_size)

    model.eval()
    with torch.no_grad():
        X_val = X_test.to(device)
        Y_val = Y_test.to(device)
        out = model(X_val)
        eval_loss = criterion(out, Y_val.squeeze().long()).item()
        test_loss.append(eval_loss)
        test_accuracy.append(accuracy_score(Y_val.cpu().squeeze(), np.argmax(out.cpu(), axis=1)))

    scheduler.step(eval_loss)
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {eval_loss:.4f}")
acc_lst.append(test_accuracy[-1])

#loss and accuracy plot
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ax[0].plot([i for i in range(num_epochs)], train_loss, label='training loss')
ax[0].plot([i for i in range(num_epochs)], test_loss, label='validation loss')
ax[0].set_title('training and validation loss')
ax[0].set_xlabel('epoch', fontsize=12)
ax[0].set_ylabel('loss', fontsize=12)
ax[0].legend()
ax[0].grid(True, linestyle='--')
ax[1].plot([i for i in range(num_epochs)], train_accuracy, label='training accuracy')
ax[1].plot([i for i in range(num_epochs)], test_accuracy, label='validation accuracy')
ax[1].set_title('training and validation accuracy')
ax[1].set_xlabel('epoch', fontsize=12)
ax[1].set_ylabel('accuracy', fontsize=12)
ax[1].legend()
ax[1].grid(True, linestyle='--')
plt.show()

#confusion heatmap
cm = confusion_matrix(Y_test, np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1), labels=[0, 1, 2, 3])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            cbar=True, annot_kws={"size": 14})
plt.suptitle("Confusion Matrix Heatmap", fontsize=16)
plt.title("0:no error; 1:oil; 2:offset; 3:Tilt", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=90)
plt.show()

#f1 score
f1 = f1_score(Y_test, np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1), average=None)
f1_df = pd.DataFrame({
    "labels":[0, 1, 2, 3],
    "f1_value":f1
})
sns.barplot(f1_df, x="labels", y="f1_value", hue="f1_value")
plt.show()

torch.save(model.cpu().state_dict(), "model.pth")
print(acc_lst)
acc_df = pd.DataFrame({
    "accuracy":acc_lst,
    "features":["I var", "rise diff", "max arc tmp", "arc stab", "weld dur", "heat speed"]
})
sns.barplot(acc_df, x="features", y="accuracy", hue="accuracy")
plt.title("accuracy when randomizing each feature")
plt.show()