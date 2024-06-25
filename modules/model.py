import torch as t
import datetime
import os

class Model(t.nn.Module):

    def train_batch(self) -> float:
        raise NotImplemented()

    def eval_batch(self) -> float:
        raise NotImplemented()

    def save(self, base_filename, dataset_name, current_update, save_dir):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_filename}_{dataset_name}.pth"
        filepath = os.path.join(save_dir, filename)
        t.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_name': dataset_name,
            'current_update': current_update,
            'timestamp': timestamp
        }, filepath)
        print(f"*******************************\nSaved record with {current_update} updates as {filename}.\n*******************************")
        return filepath

    def load(self, fn):
        if not os.path.exists(fn):
            return -1
        record = t.load(fn, map_location=t.device(self.device))
        self.batch_idx = record["batch_idx"]
        self.load_state_dict(record["model_state_dict"])
        self.optimizer.load_state_dict(record["optimizer_state_dict"])
        return record["current_update"]
