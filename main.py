import glob

import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from model import LSTM_Model
from SAnD import SAnD
from readers import PhenotypingReader
from parser import parse
from utils import *
from preprocessing import *
from batch_gen import BatchGen

args = parse()

args.gpu = get_free_gpu()
torch.cuda.set_device(args.gpu)

if not os.path.exists("./exp/"):
    os.mkdir("./exp/")
experiment_path = f"./exp/{args.dst_folder}"
create_exp_dir(experiment_path, scripts_to_save=glob.glob("*.py"))

kwargs = {"num_workers": 5, "pin_memory": True} if args.gpu else {}

train_reader = PhenotypingReader(
    dataset_dir=os.path.join(args.data, "train"),
    listfile=os.path.join(args.data, "train_listfile.csv"),
)
val_reader = PhenotypingReader(
    dataset_dir=os.path.join(args.data, "train"),
    listfile=os.path.join(args.data, "val_listfile.csv"),
)

discretizer = Discretizer(
    timestep=1.0,  # time step
    store_masks=True,
    impute_strategy="previous",
    start_time="zero",
)

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(
    ","
)
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = "./ph_ts1.0.input_str:previous.start_time:zero.normalizer"
normalizer.load_params(normalizer_state)

train_data_gen = BatchGen(
    train_reader,
    discretizer,
    normalizer,
    args.train_batch_size,
    False,
    False,
    shuffle=True,
)

test_data_gen = BatchGen(
    val_reader,
    discretizer,
    normalizer,
    args.test_batch_size,
    False,
    False,
    shuffle=False,
)

if args.model_type == "LSTM":
    model = LSTM_Model(args)
if args.model_type == "SAnD":
    model = SAnD(
        input_features=76,
        seq_len=args.sequence_length,
        n_heads=16,
        factor=1,
        n_class=25,
        n_layers=2,
        d_model=256,
        dropout_rate=args.dropout_rate,
    )

model = model.cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.98),
)

crit = nn.BCEWithLogitsLoss()

loss_mtr = AverageMeter()
auroc_micro_mtr = MetricMeter(ROCAUC("micro"))
auroc_macro_mtr = MetricMeter(ROCAUC("macro"))

for epoch in range(args.epochs):
    model.train()
    loss_mtr.reset()
    auroc_micro_mtr.reset()
    auroc_macro_mtr.reset()

    for batch_idx in range(len(train_data_gen)):
        data = next(train_data_gen)
        seq_lens = data[2]
        label = data[1]
        data = data[0]

        if args.gpu:
            data = data.cuda()
            label = label.cuda()

        output = model(data, seq_lens)

        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        label_tmp = label.cpu().numpy()
        loss_mtr.update(loss.item(), data.size(0))
        target = torch.argmax(label, dim=1)
        pred = torch.argmax(output, dim=1)
        output = output.cpu().detach().numpy()

        target = target.cpu().numpy()
        pred = pred.cpu().detach().numpy()

        auroc_micro_mtr.update(label_tmp, output)
        auroc_macro_mtr.update(label_tmp, output)

        if (batch_idx + 1) % args.report_freq == 0:
            print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}")

    print(f"Train: epoch: {epoch+1}, loss = {loss_mtr.avg}")
    print(
        f"Train: epoch: {epoch+1}, AUROC Micro = {auroc_micro_mtr.score()}, AUROC Macro = {auroc_macro_mtr.score()}"
    )

    model.eval()
    loss_mtr.reset()
    auroc_micro_mtr.reset()
    auroc_macro_mtr.reset()
    with torch.no_grad():
        for batch_idx in range(len(test_data_gen)):
            data = next(train_data_gen)
            seq_lens = data[2]
            label = data[1]
            data = data[0]

            if args.gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data, seq_lens)
            loss = crit(output, label)

            label_tmp = label.cpu().numpy()
            loss_mtr.update(loss.item(), data.size(0))
            target = torch.argmax(label, dim=1)
            pred = torch.argmax(output, dim=1)
            output = output.cpu().detach().numpy()

            target = target.cpu().numpy()
            pred = pred.cpu().detach().numpy()

            auroc_micro_mtr.update(label_tmp, output)
            auroc_macro_mtr.update(label_tmp, output)

            if (batch_idx + 1) % args.report_freq == 0:
                print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}")

        print(f"Eval: epoch: {epoch+1}, loss = {loss_mtr.avg}")
        print(
            f"Eval: epoch: {epoch+1}, AUROC Micro = {auroc_micro_mtr.score()}, AUROC Macro = {auroc_macro_mtr.score()}"
        )
