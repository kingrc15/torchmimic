import glob

import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from model import LSTM_Model
from data import PhenotypingReader
from parser import parse
from utils import *
from preprocessing import *

args = parse()

torch.cuda.set_device(args.gpu)

if not os.path.exists("./exp/"):
    os.mkdir("./exp/")
experiment_path = f"./exp/{args.dst_folder}"
create_exp_dir(experiment_path, scripts_to_save=glob.glob("*.py"))

kwargs = {"num_workers": 5, "pin_memory": True} if args.gpu else {}

train_data = PhenotypingReader("/data/datasets/mimic3-benchmarks/data/phenotyping", "train_listfile.csv")
test_data = PhenotypingReader("/data/datasets/mimic3-benchmarks/data/phenotyping", "test_listfile.csv")

discretizer = Discretizer(timestep=1., # time step
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_data.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = './ph_ts1.0.input_str:previous.start_time:zero.normalizer'
normalizer.load_params(normalizer_state)

train_data_gen = BatchGen(train_data, discretizer,
                        normalizer, args.train_batch_size,
                        False, False, shuffle=True)

test_data_gen = BatchGen(test_data, discretizer,
                        normalizer, args.train_batch_size,
                        False, False, shuffle=False)

if args.model_type == "LSTM":
    model = LSTM_Model()
    
optimizer = optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)  

crit = nn.BCELoss()

loss_mtr = AverageMeter()

for epoch in range(args.epochs):
    loss_mtr.reset()
    for batch_idx, (data, label) in enumerate(train_data_gen):
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        
        if args.gpu:
            data = data.cuda()
            label = label.cuda()
        
        output = model(data)
        
        # print(output.size(), label.size())
        loss = crit(output[:,-1,:], label)
        loss.backward()
        optimizer.step()
        
        loss_mtr.update(loss.item(), data.size(0))
        
        if (batch_idx + 1) % args.report_freq == 0:
            print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}")
            
    print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}")
            
    model.eval()
    loss_mtr.reset()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_data_gen):
            if args.gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data)
            loss = crit(output, label)

            loss_mtr.update(loss.item(), data.size(0))

            if (batch_idx + 1) % args.report_freq == 0:
                print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}")
                
        print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}")
                
    model.train()
