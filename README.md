# MIMIC Benchmark in PyTorch

MIMIC Benchmark reimplemented in PyTorch. 

## Phenotyping 

LSTM
`python main.py --model_type LSTM --train_batch_size 8 --learning_rate 0.001 --bidirectional --weight_decay 0 --dropout_rate 0.3`


## TODOs

### Benchmarks

- [x] Phenotyping
- [ ] Length of Stay
- [ ] Decompensation
- [ ] In Hospital Mortality
- [ ] Multitask

### Models
- [x] Standard LSTM
- [ ] Standard LSTM + Deep Supervision
- [ ] Logistic Regression
- [ ] Channelwise LSTM
- [ ] Channelwise LSTM + Deep Supervision
- [ ] Multitask LSTM
- [ ] Multitask Channelwise LSTM

## Testing

`python -m pytest ./tests/*.py -s`