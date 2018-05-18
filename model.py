import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import UNK_ID


class DropWord(nn.Module):
    def __init__(self, dropout, unk_id):
        super().__init__()
        self.dropout = dropout
        self.unk_id = unk_id

    def forward(self, inputs):
        if not self.training or self.dropout == 0:
            return inputs
        else:
            dropmask = torch.bernoulli(
                torch.full(inputs.size(), self.dropout, dtype=torch.float())).byte()
            inputs = inputs.clone()
            inputs[dropmask] = self.unk_id
            return inputs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, inputs, lengths):
        inputs = self.drop(inputs)
        inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        _, (hn, _) = self.rnn(inputs)
        return hn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fcz = nn.Linear(code_size, hidden_size * 2)
        
    def forward(self, inputs, z, lengths=None, init_hidden=None):
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        if init_hidden is None:
            init_hidden = [x.contiguous() for x in torch.chunk(F.tanh(self.fcz(z)), 2, 2)]
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, code_size, dropout, dropword):
        super().__init__()
        self.dropword = DropWord(dropword, UNK_ID)
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size, code_size, dropout)
        self.fcmu = nn.Linear(hidden_size, code_size)
        self.fclogvar = nn.Linear(hidden_size, code_size)
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.bow_predictor = BowPredictor(code_size, hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, lengths):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(self.dropword(inputs))
        hn = self.encoder(enc_emb, lengths)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        outputs, _ = self.decoder(dec_emb, z, lengths=lengths)
        outputs = self.fcout(outputs)
        bow = self.bow_predictor(z)
        return outputs, mu, logvar, bow

    def reconstruct(self, inputs, lengths, max_length, sos_id):
        enc_emb = self.lookup(inputs)
        hn = self.encoder(enc_emb, lengths)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        # z size: 1 x batch_size x code_size
        z = self.reparameterize(mu, logvar)
        return self.generate(z, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id, device):
        z_size = self.fcmu.out_features
        z = torch.randn(1, num_samples, z_size, device=device)
        return self.generate(z, max_length, sos_id)
    
    def generate(self, z, max_length, sos_id):
        batch_size = z.size(1)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = None
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, z, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        return self.fc2(self.act(self.fc1(inputs)))


class BowPredictor(MLP):
    def __init__(self, code_size, hidden_size, vocab_size):
        super().__init__(code_size, hidden_size, vocab_size)

    def forward(self, inputs):
        """Inputs: latent code """
        return super().forward(inputs).squeeze(0)


class ZPrior(MLP):
    def __init__(self, input_size, hidden_size, code_size):
        super().__init__(input_size, hidden_size, code_size * 2)

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs.unsqueeze_(0)
        return torch.chunk(super().forward(inputs), chunks=2, dim=2)


class TextCVAE(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_size, label_embed_size,
                 hidden_size, code_size, dropout, dropword):
        super().__init__()
        self.dropword = DropWord(dropword, UNK_ID)
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.label_lookup = nn.Embedding(num_classes, label_embed_size)
        self.encoder = Encoder(embed_size, hidden_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size,
                               code_size + label_embed_size, dropout)
        self.fcmu = nn.Linear(hidden_size + label_embed_size, code_size)
        self.fclogvar = nn.Linear(hidden_size + label_embed_size, code_size)
        self.z_prior = ZPrior(label_embed_size, hidden_size, code_size)
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.bow_predictor = BowPredictor(code_size + label_embed_size,
                                          hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, labels, lengths):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(self.dropword(inputs))
        lab_emb = self.label_lookup(labels).unsqueeze(0)
        mu_pr, logvar_pr = self.z_prior(lab_emb)
        hn = self.encoder(enc_emb, lengths)
        hn = torch.cat([hn, lab_emb], dim=2)
        mu_po, logvar_po = self.fcmu(hn), self.fclogvar(hn)
        if self.training:
            z = self.reparameterize(mu_po, logvar_po)
        else:
            z = mu_po
        code = torch.cat([z, lab_emb], dim=2)
        outputs, _ = self.decoder(dec_emb, code, lengths=lengths)
        outputs = self.fcout(outputs)
        bow = self.bow_predictor(code)
        return outputs, (mu_pr, mu_po), (logvar_pr, logvar_po), bow

    def reconstruct(self, inputs, labels, lengths, max_length, sos_id):
        enc_emb = self.lookup(inputs)
        lab_emb = self.label_lookup(labels).unsqueeze(0)
        hn = self.encoder(enc_emb, lengths)
        hn = torch.cat([hn, lab_emb], dim=2)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        # z size: 1 x batch_size x code_size
        z = self.reparameterize(mu, logvar)
        return self.generate(z, lab_emb, max_length, sos_id)

    def sample(self, labels, max_length, sos_id):
        lab_emb = self.label_lookup(labels).unsqueeze(0)
        mu, logvar = self.z_prior(lab_emb)
        z = self.reparameterize(mu, logvar)
        return self.generate(z, lab_emb, max_length, sos_id)
    
    def generate(self, z, lab_emb, max_length, sos_id):
        batch_size = z.size(1)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        code = torch.cat([z, lab_emb], dim=2)
        hidden = None
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, code, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    
class LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.fcout = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, lengths):
        dec_emb = self.drop(self.lookup(inputs))
        if lengths is not None:
            dec_emb = pack_padded_sequence(dec_emb, lengths, batch_first=True)
        outputs, hidden = self.rnn(dec_emb)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.fcout(self.drop(outputs))
        return outputs


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        if inputs.dim() == 3:
            inputs = inputs.squeeze(0)
        return F.log_softmax(self.fc2(self.activation(self.fc1(inputs))), dim=1)


class SSTextVAE(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_size, y_embed_size, hidden_size, code_size, dropout):
        super(SSTextVAE, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.y_lookup = nn.Embedding(num_classes, y_embed_size)
        self.encoder = Encoder(embed_size, hidden_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size, code_size + y_embed_size, dropout)
        self.fcmu = nn.Linear(hidden_size + y_embed_size, code_size)
        self.fclogvar = nn.Linear(hidden_size + y_embed_size, code_size)
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.bow_predictor = BoWPredictor(vocab_size, hidden_size, code_size + y_embed_size)
        self.classifier = Classifier(hidden_size, hidden_size, num_classes)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, lengths, temp=None, y=None):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(inputs)
        hn = self.encoder(enc_emb, lengths)
        py = self.classifier(hn)
        if y is None:
            dist = RelaxedOneHotCategorical(temp, logits=py)
            y = dist.sample().max(1)[1]
        y_emb = self.y_lookup(y)
        h = torch.cat([hn, y_emb.unsqueeze(0)], dim=2)
        mu, logvar = self.fcmu(h), self.fclogvar(h)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        code = torch.cat([z, y_emb.unsqueeze(0)], dim=2)
        outputs, _ = self.decoder(dec_emb, code, lengths=lengths)
        outputs = self.fcout(outputs)
        bow = self.bow_predictor(code)
        return outputs, mu, logvar, bow, py

