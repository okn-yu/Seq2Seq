import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.data_gen import subtraction_datasets, to_ID, to_batch
from src.endeco import Encoder, Decoder

embedding_dim = 200
hidden_dim = 128
vocab_size = len(to_ID())

BATCH_NUM = 100
EPOCH_NUM = 100

device = torch.device("cuda" if torch.cudnn_is_acceptable() else "cpu")
encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

criterion = nn.CrossEntropyLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.paparameters(), lr=0.001)

input_data, output_data = subtraction_datasets()
train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size=0.7)

all_losses = []
print("training...")

for epoch in range(1, EPOCH_NUM + 1):
    epoch_loss = 0

    input_batch, output_batch = to_batch(train_x, train_y, batch_size=BATCH_NUM)

    for i in range(len(input_batch)):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_ts = torch.tensor(input_batch[i], device=device)
        output_ts = torch.tensor(output_batch[i], device=device)

        encoder_state = encoder(input_ts)

        # last-data can not be next-input.
        training_source = output_ts[:, :-1]
        # start symbol can not ba data to train.
        training_data = output_ts[:, 1:]

        loss = 0

        decoder_output, _ = decoder(training_source, encoder_state)

        for j in range(decoder_output.size()[1]):
            loss += criterion(decoder_output[:, j, :], training_data[:, j])

        epoch_loss += loss.item()
        loss.backward()

        encoder_optimizer.step()
        decoder_output.step()

    print("Epoch %d: %.2f" % (epoch, epoch_loss))
    all_losses.append(epoch_loss)
    if epoch_loss < 1: break
print("Done")
