# MINST Model

## Model


### r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:26, j_out:1 
self.conv1 = nn.Conv2d(1, 16, 3, padding=0)
self.bn1 = nn.BatchNorm2d(16)
### r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out:24, j_out:1
self.conv2 = nn.Conv2d(16, 16, 3) # output_size = 24 #rf 5
self.bn2 = nn.BatchNorm2d(16)
### r_in:5, n_in:24, j_in:1, s:1, r_out:7, n_out:22, j_out:1
self.conv3 = nn.Conv2d(16, 16, 3) # output_size = 22 #rf 7
self.bn3 = nn.BatchNorm2d(16)
### r_in:7, n_in:24, j_in:1, s:2, r_out:8, n_out:11, j_out:2
self.pool1 = nn.MaxPool2d(2, 2)
self.dp1 = nn.Dropout2d(0.1)
### r_in:8, n_in:11, j_in:2, s:1, r_out:12, n_out:9, j_out:2
self.conv4 = nn.Conv2d(16, 16, 3)
self.bn5 = nn.BatchNorm2d(16)
### r_in:12, n_in:9, j_in:2, s:1, r_out:16, n_out:7, j_out:2
self.conv5 = nn.Conv2d(16, 16, 3)
self.bn6 = nn.BatchNorm2d(16)
self.dp2 = nn.Dropout2d(0.1)
### r_in:16, n_in:7, j_in:2, s:1, r_out:20, n_out:1, j_out:2
self.conv6 = nn.Conv2d(16, 10, 7)


## No parameters: 17,450
## Epochs: 20
## Accuracy: 99.42%
