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
    
if args.gpu:
    model = model.cuda()
    
optimizer = optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)  

crit = nn.BCELoss()

loss_mtr = AverageMeter()
acc_mtr = MetricMeter(accuracy)
auroc_micro_mtr = MetricMeter(ROCAUC('micro'))
auroc_macro_mtr = MetricMeter(ROCAUC('macro'))

for epoch in range(args.epochs):
    loss_mtr.reset()
    acc_mtr.reset()
    auroc_micro_mtr.reset()
    auroc_macro_mtr.reset()
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
        target = torch.argmax(label, dim=1)
        pred = torch.argmax(output[:,-1,:], dim=1)

        target = target.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        acc_mtr.update(target, pred)
        
        output = output[:,-1,:].cpu().detach().numpy()
        auroc_micro_mtr.update(target, output)
        auroc_macro_mtr.update(target, output)
        
        if (batch_idx + 1) % args.report_freq == 0:
            print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}")
            
    print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}, accuracy = {acc_mtr.score()}")
    print(f"Train: epoch: {epoch+1}, AUROC Micro = {auroc_micro_mtr.score()}, AUROC Macro = {auroc_macro_mtr.score()}")
            
    model.eval()
    loss_mtr.reset()
    acc_mtr.reset()
    auroc_micro_mtr.reset()
    auroc_macro_mtr.reset()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_data_gen):
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
        
            if args.gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data)
            loss = crit(output[:,-1,:], label)

            loss_mtr.update(loss.item(), data.size(0))
            target = torch.argmax(label, dim=1)
            pred = torch.argmax(output[:,-1,:], dim=1)

            target = target.cpu().numpy()
            pred = pred.cpu().detach().numpy()
            acc_mtr.update(target, pred)

            output = output[:,-1,:].cpu().detach().numpy()
            auroc_micro_mtr.update(target, output)
            auroc_macro_mtr.update(target, output)

            if (batch_idx + 1) % args.report_freq == 0:
                print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}")
                
        print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}, accuracy = {acc_mtr.score()}")
        print(f"Eval: epoch: {epoch+1}, AUROC Micro = {auroc_micro_mtr.score()}, AUROC Macro = {auroc_macro_mtr.score()}")
                
    model.train()
