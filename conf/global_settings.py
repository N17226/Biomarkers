
import os
from datetime import datetime

CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH=100
EPOCH1=5

MILESTONES = [10, 30, 40,101]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10






