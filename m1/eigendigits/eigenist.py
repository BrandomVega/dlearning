import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt; plt.style.use('bmh')
import numpy as np
import math

# MNIST digits
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='E:/datasets', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='E:/datasets', train=False, download=True, transform=transform)

# Dataset
dataset = torch.stack([matrix for matrix, label in mnist_train])
samples = len(dataset)
test_samples = len(mnist_test)
n,n = dataset[0][0].shape
print(samples,n,n)

# Dataset mean
mean = torch.zeros_like(dataset[0][0])
for m in range(samples):
    mean += dataset[m][0]
mean /= samples

# Subtract mean from dataset
for m in range(samples):
    dataset[m][0] = mean - dataset[m][0]

plt.imshow(dataset[321][0], cmap='hsv')
plt.title("MNIST avg - image")
plt.colorbar()
plt.show()

# Covariance matrix
omega_shape = torch.flatten(dataset[0][0]).shape[0] 
C = torch.zeros((omega_shape, omega_shape), dtype=torch.float32)
for m in range(samples):
    omega = torch.flatten(dataset[m][0]).unsqueeze(1) 
    C += omega @ omega.T  
C /= samples

plt.imshow(C)
plt.colorbar()
plt.title("Covariance Matrix")
plt.show()

# Eigenvalues 
eigenvalues, eigenvectors = np.linalg.eig(C)

k=0
for i in range(len(eigenvalues)):
    var_explained = sum(eigenvalues[:i])/ eigenvalues.sum()
    #print(f"var: {var_explained} at {i}")
    if var_explained > 0.90:
        k=i
        break
k = len(eigenvalues)

# Calculate avg of samples and map to eigenspace with omega_k for for each k digit class
digits = []
dataset_samples = 10000000000
for l in range(10):
    print("Calculando ",l)
    actual = []
    # omega sample avg
    samples = []
    for image,label in mnist_train:
        if label == l:
            samples.append([image,label])  
            if len(samples) == dataset_samples:
                break    
    avg = torch.mean(torch.stack([x[0] for x in samples]), dim=0)
    for i in range(k):
        actual.append((eigenvectors[i].T@(torch.flatten(avg[0]))).item())
    digits.append(actual)

# Test model
# Calculate the distance of new image with representative omega_k
top3acc = 0
top1acc = 0
for d,(image,label) in enumerate(mnist_test):
    print(f"{d}/{test_samples}")
    # image:omega
    # Tenía mal el label. Aún así las distancias son muy parecidas
    omega = []
    for i in range(k):
        w_i = (eigenvectors[i].T@(torch.flatten(mean - image))).item()
        omega.append(w_i)
    #print(omega)
    # Distances from the omega_avg
    distances = []
    #print(label)
    for l in range(10):
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(omega,  digits[l])))
        distances.append((l,dist))
        #print(f"Distance {l} = {dist}")

    distances = sorted(distances, key = lambda x:x[1], reverse = True)
    #print(distances)
    #_ = input("continuar ")
    if distances[0][0] == label:
        top1acc+=1
    if distances[0][0] == label or distances[1][0] == label or distances[2][0] == label:
        top3acc+=1

print(f"Top 1 Accuracy {top1acc}/{test_samples} = {top1acc/test_samples}")
print(f"Top 3 Accuracy {top3acc}/{test_samples} = {top3acc/test_samples}")
