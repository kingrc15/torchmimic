# MIMIC Benchmark in PyTorch

## Phenotyping 

LSTM
`python main.py --model_type LSTM --train_batch_size 8 --learning_rate 0.001 --bidirectional --weight_decay 0 --dropout_rate 0.3`

SAnD
`python main.py --model_type SAnD --train_batch_size 128 --learning_rate 0.0005 --dropout_rate 0.4 --hidden_dim 256, num_layers 2`