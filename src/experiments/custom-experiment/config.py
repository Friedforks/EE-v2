# configuration file

# data
# base_dir="C:\\WorkDir\\ComputerScience\\MachineLearning\\EE\\src"
base_dir="/mnt/c/WorkDir/ComputerScience/MachineLearning/EE/src"
lookback=128
pred_length=128

# training and model
batch_size = 128
epochs = 10
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.1
num_workers=8

# transformer
hidden_dim = 8
num_heads=1
transformer_layers = 4
ffn_dim = 16

