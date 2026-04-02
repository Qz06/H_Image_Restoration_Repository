from models.model_plain import ModelPlain
import torch


class ModelPlain4(ModelPlain):
    """Train with four inputs (L, k, sf, sigma) and with pixel loss for USRNet"""

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)  # low-quality image
        self.k = data['k'].to(self.device)  # blur kernel
        sf = data['sf']
        if torch.is_tensor(sf):
            self.sf = sf.view(-1).to(dtype=torch.int64)
        else:
            self.sf = torch.tensor([int(sf)], dtype=torch.int64)
        self.sigma = data['sigma'].to(self.device)  # noise level
        if need_H:
            self.H = data['H'].to(self.device)  # H

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        sf_values = self.sf.tolist() if torch.is_tensor(self.sf) else [int(self.sf)]
        if len(set(sf_values)) == 1:
            self.E = self.netG(self.L, self.k, int(sf_values[0]), self.sigma)
            return

        outputs = []
        for i, sf in enumerate(sf_values):
            outputs.append(self.netG(self.L[i:i+1], self.k[i:i+1], int(sf), self.sigma[i:i+1]))
        self.E = torch.cat(outputs, dim=0)
