
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

args.n_classes = 25

args.gpu = get_free_gpu()
torch.cuda.set_device(args.gpu)

log = Logger(args)

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
    shuffle=False,
    return_mask=args.model_type=="SAnD"
)

test_data_gen = BatchGen(
    val_reader,
    discretizer,
    normalizer,
    args.test_batch_size,
    False,
    False,
    shuffle=False,
    return_mask=args.model_type=="SAnD"
)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.gpu else {}

train_loader = DataLoader(
      train_data_gen, batch_size=args.train_batch_size, shuffle=True, collate_fn=pad_colalte, **kwargs)
test_loader = DataLoader(
      test_data_gen, batch_size=args.test_batch_size, shuffle=False, collate_fn=pad_colalte, **kwargs)

if args.model_type == "LSTM":
    model = LSTM_Model(args)
if args.model_type == "SAnD":
    model = SAnD(
        input_features=76,
        seq_len=train_data_gen.get_max_seq_length(),
        n_heads=args.n_heads,
        factor=args.factor,
        n_class=args.n_classes,
        n_layers=args.num_layers,
        d_model=args.hidden_dim,
        dropout_rate=args.dropout_rate,
    )

model = model.cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.98),
)

crit = nn.BCELoss()

loss_mtr = AverageMeter()
auroc_micro_mtr = MetricMeter(ROCAUC("micro"))
auroc_macro_mtr = MetricMeter(ROCAUC("macro"))

for epoch in range(args.epochs):
    model.train()
    log.reset()
    for batch_idx, (data, label, lens, mask) in enumerate(train_loader):
        if args.gpu:
            data = data.cuda()
            label = label.cuda()

        output = model((data, lens))
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        log.update(output, label, loss)

        if (batch_idx + 1) % args.report_freq == 0:
            print(f"Train: epoch: {epoch+1}, loss = {log.get_loss()}")

    log.print_metrics(epoch,split="Train")

    model.eval()
    log.reset()
    with torch.no_grad():
        for batch_idx, (data, label, lens, mask) in enumerate(test_loader):
            if args.gpu:
                data = data.cuda()
                label = label.cuda()

            output = model((data, lens))
            loss = crit(output, label)

            log.update(output, label, loss)

            if (batch_idx + 1) % args.report_freq == 0:
                print(f"Eval: epoch: {epoch+1}, loss = {log.get_loss()}")

        log.print_metrics(epoch,split="Eval")
