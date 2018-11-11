#params
from datetime import datetime
import argparse



DATA_DIR = '/home/comp/chongyin/DataSets/DA/Office-31/Original_images'
LIST1_PATH = '/home/comp/chongyin/TensorFlow/DeepDICD/OFFICE/data/list/amazon.txt' 
LIST2_PATH = '/home/comp/chongyin/TensorFlow/DeepDICD/OFFICE/data/list/dslr.txt' 
PRETRAINED_MODEL='/home/comp/chongyin/TensorFlow/DeepDICD/OFFICE/pretrained/bvlc_alexnet.npy'

MOMENTUM = 0.9
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001 #chong
SAVE_NUM_IMAGES = 4
TEST_EVERY = 50
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
SNAPSHOT_DIR = '/home/comp/chongyin/checkpoints/DeepDICD/'+subdir+'/'



def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument('--h',type=int,default=227)
    parser.add_argument('--w',type=int,default=227)
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--power", type=float, default=POWER)
    parser.add_argument('--learning_strategy',type=int,default=3)
    
    
    parser.add_argument("--test_every", type=int, default=TEST_EVERY)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--no_update_mean_var", action="store_true")
    parser.add_argument("--train_beta_gamma", action="store_true")

    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--restore_from", type=str, default=PRETRAINED_MODEL)
    parser.add_argument('--not_load_pretrained',action='store_true')
    parser.add_argument('--img_dir',type=str, default=DATA_DIR)
    parser.add_argument('--list1',type=str,default=LIST1_PATH)
    parser.add_argument('--list2',type=str,default=LIST2_PATH)
    parser.add_argument('--domain',type=str,default='A-W')
    parser.add_argument('--model_name',type=str,default='Base')
    parser.add_argument('--start_steps',type=int,default=0)

    parser.add_argument('--num_epochs',type=int,default=4000)
    parser.add_argument('--num_threads',type=int,default=32)
    parser.add_argument('--encoder', type=str, default='alexnet')
    parser.add_argument('--drop_rate',type=float,default=0.5)
    parser.add_argument('--num_classes',type=int,default=31)

    parser.add_argument('--num_gpus',type=int, default=1)

    parser.add_argument('--resume_from',type=str,default='')
   

    return parser.parse_args()


