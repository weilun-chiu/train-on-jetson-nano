import os
import time
import random
import shutil
import librosa
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

train_dir = './Train'
valid_dir = './Valid'

spectrogram_length = 395
feature_size = 513
emotion2idx_dict = {'ANG': 0, 'NEU': 1, 'DIS': 2, 'SAD': 3, 'FEA': 4, 'HAP': 5}
idx2emotion_dict = {0: 'ANG', 1: 'NEU', 2: 'DIS', 3: 'SAD', 4: 'FEA', 5: 'HAP'}


class Speech_Dataset(Dataset):
    def __init__(self, data_dir, emotion2idx_dict, spectrogram_length):
        self.data_dir = data_dir
        self.emotion2idx_dict = emotion2idx_dict
        self.spectrogram_length = spectrogram_length

        self.filenames = os.listdir(data_dir)
        self.labels = [self.emotion2idx_dict[filename.split('_')[2]] for filename in self.filenames]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]

        # load the speech file and calculate spectrogram
        speech_audio, _ = librosa.load(os.path.join(self.data_dir, filename), sr = 16000 * 0.8)
        spectrogram = librosa.stft(speech_audio, n_fft=1024, hop_length=160, center=False, win_length=1024)
        spectrogram = abs(spectrogram)
        
        feature_size, length = spectrogram.shape

        # modify the length of the spectrogram to be the same as the specified length
        if length > self.spectrogram_length:
            spectrogram = spectrogram[:, :self.spectrogram_length]
        else:
            cols_needed = self.spectrogram_length - length
            spectrogram = np.concatenate((spectrogram, np.zeros((feature_size, cols_needed))), axis=1)
        
        """
        # Calculate the new dimensions (80% of the original size)
        new_width = int(spectrogram.shape[1] * 0.8)
        new_height = int(spectrogram.shape[0] * 0.8)

        # Resize the array using the resize function with interpolation
        spectrogram = np.resize(spectrogram, (new_height, new_width))
        """
        
        return np.expand_dims(spectrogram.astype(np.float32), axis=0) , label
    
# create training and validation datasets and dataloaders
train_dataset = Speech_Dataset(train_dir, emotion2idx_dict, spectrogram_length)
valid_dataset = Speech_Dataset(valid_dir, emotion2idx_dict, spectrogram_length)

train_dataloader = DataLoader(train_dataset, batch_size=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Linear layer with reduced precision (FP16)
        self.fc1 = nn.Linear(16 * 64 * 49, 64)
        #.to(torch.float16)
        
        
        # The last linear layer, operates in FP32
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 16 * 64 * 49)
        #x = x.to(torch.float16)
        x = self.fc1(x)
        #x = x.to(torch.float32)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# create directory for saving the checkpoint files
checkpoint_dir = './checkpoint'
#os.makedirs(checkpoint_dir)

num_classes = len(emotion2idx_dict)
num_epochs = 2

# Remember to change the output filename and model
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f'using device {device}')
output_filename = os.path.join(checkpoint_dir, 'SpectrogramCNN.pth')
model = SpectrogramCNN(num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training script
start_time = time.time()

for epoch in range(num_epochs):
    #model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs.to(device)
        labels.to(device)
        outputs = model(inputs)
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.2f}%')

end_time = time.time()

print('Training Latency in seconds: {}'.format(round(end_time - start_time, 3)))

torch.save(model.state_dict(), output_filename)

# load model
model = Sample_Rate_80_Percent_SpectrogramCNN(num_classes)
model.load_state_dict(torch.load(output_filename))
model.eval()

# calculate total number of parameters in the model 
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter Count: {total_params}")

# calculate FLOPs count for a single inference
channel = 1
height = feature_size
width = spectrogram_length
kernel_size = 3

FLOPs_count = 0

for name, param in model.named_parameters():
    param_type = name.split('.')[1]

    if name.startswith('conv'):
        if param_type ==  'weight':
            out_channel, in_channel, kernel_size, kernel_size = param.size()
            mul_FLOPs = kernel_size * kernel_size * in_channel
            add_FLOPs = (kernel_size * kernel_size - 1) * in_channel + (in_channel - 1)
            single_kernel_FLOPs = mul_FLOPs + add_FLOPs
            FLOPs_count += single_kernel_FLOPs * height * width * out_channel
            channel = out_channel
        elif param_type == 'bias':
            FLOPs_count += channel * height * width
    elif name.startswith('bn'):
        FLOPs_count += channel * height * width
        if param_type == 'bias':
            height = height // 2
            width = width // 2
    elif name.startswith('fc'):
        if param_type ==  'weight':
            out_neuron, in_neuron = param.size()
            FLOPs_count += (2 * in_neuron - 1) * out_neuron
        elif param_type == 'bias':
            FLOPs_count += param.size()[0]

print("Total FLOPs Count: {}".format(FLOPs_count))

# validation script
correct_predictions = 0
total_samples = 0
true_labels = []
predicted_labels = []

start_time = time.time()

with torch.no_grad():
    for inputs, labels in tqdm(valid_dataloader):
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

end_time = time.time()

accuracy = correct_predictions / total_samples * 100
print(f'Validation Accuracy: {accuracy:.2f}%')

print('Average Inference Latency in seconds: {}'.format(round((end_time - start_time) / len(valid_dataloader), 6)))
