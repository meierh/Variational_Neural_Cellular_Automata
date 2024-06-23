import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

#modify
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_dir)

#original revision = os.environ.get("REVISION") or "%s" % datetime.now()
revision = os.environ.get("REVISION") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#original message = os.environ.get('MESSAGE')
message = os.environ.get('MESSAGE') or ""
#original tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or "/tmp/tensorboard"
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or os.path.join(project_root_dir, "tensorboard")
flush_secs = 10


#original def get_writers(name):
#    train_writer = SummaryWriter(tensorboard_dir + '/%s/tensorboard/%s/train/%s' % (name, revision, message), flush_secs=flush_secs)
#    test_writer = SummaryWriter(tensorboard_dir + '/%s/tensorboard/%s/test/%s' % (name, revision, message), flush_secs=flush_secs)
#    return train_writer, test_writer

def get_writers(name):
    train_writer = SummaryWriter(os.path.join(tensorboard_dir, name, 'train', revision, message), flush_secs=flush_secs)
    test_writer = SummaryWriter(os.path.join(tensorboard_dir, name, 'test', revision, message), flush_secs=flush_secs)
    return train_writer, test_writer
