import tqdm
import os

from modules.model import Model

def train(model: Model, dataset_name, n_updates, eval_interval, checkpoint_path="latest", save_dir="results"):
    best = float("inf")
    start_iter = 0

    if os.path.exists(checkpoint_path):
        start_iter = model.load(checkpoint_path)

    for i in tqdm.tqdm(range(start_iter, n_updates)):
        model.train_batch()
        
        if (i + 1) % 50 == 0:
            save_filename = model.save("checkpoint", dataset_name, i + 1, save_dir)

        if (i + 1) % eval_interval == 0:
            loss = model.eval_batch()
            latest_filename = model.save("latest", dataset_name, i + 1, save_dir)
            if loss < best:
                best = loss
                best_filename = model.save("best", dataset_name, i + 1, save_dir)
