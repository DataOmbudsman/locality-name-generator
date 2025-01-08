from datetime import datetime
from pathlib import Path

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, trainer_config):
        self._set_up_logging()
        self.model = model
        self.criterion = criterion
        self.config = trainer_config

    def _set_up_logging(self):
        dt = datetime.now().strftime("%Y%m%d%H%M")
        log_path = Path("logs") / dt
        writer = SummaryWriter(log_path)
        self.writer = writer

    def train_step(self, x, y, lengths):
        self.model.train()
        prev_state = self.model.init_state(
            batch_size=x.shape[0], device=self.config.device
        )
        out, state = self.model(x, lengths, prev_state)
        loss = self.criterion(out.transpose(1, 2), y)
        return loss

    def train(self, data, print_every=50):
        loss_history = []
        running_loss = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        step = 0
        for epoch in range(1, self.config.n_epochs + 1):
        
            for x, y, lengths in data:

                loss = self.train_step(x, y, lengths)
                loss.backward()

                loss_history.append(loss.item())
                running_loss += loss.item()

                clip = self.config.clip
                if clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip
                    )

                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                if print_every and (step % print_every == 0):
                    self._log_at_step(
                        running_loss, loss, epoch, step, print_every
                    )
                    running_loss = 0

        return self.model, loss_history

    def _log_at_step(self, running_loss, loss, epoch, step, print_every):
        avg_loss = running_loss / print_every
        print(f"Epoch: {epoch}, Iteration: {step}, Loss: {avg_loss:.4f}")      
        self.writer.add_scalar(f"loss", loss, step)
