import random
import numpy as np
from tqdm import tqdm
import torch
import ujson
from data_loader import DatasetIterator
from vgg import vgg16_bn
from train import Trainer
from mylogger import create_logger
logger = create_logger(__name__,"/home/cuiz/logs/",to_disk=True, prefix='piying')
class Config():
	def __init__(self) -> None:
		#path setting
		self.root_path = "/home/cuiz/图片分类/"
		#"/Users/cz/Desktop/KnowledgeAndCareer/项目和代码/图片分类/"
		self.train_path = self.root_path + "train_dataset.json"
		self.dev_path = self.root_path + "dev_dataset.json"
		self.test_path = self.root_path + "test_dataset.json"
		self.picture_path = self.root_path + "PiYingImage_v3cp"
		
		#save setting
		self.task_name = "piying_classify"

		#model setting
		self.hidden_size = 512
		self.label_num = 5
		self.momentum= 0.9
		self.weight_decay = 5e-4
		#training setting
		self.batch_size = 64
		self.epochs = 50
		self.device_ids = [0,1,2,3]
		self.learning_rate = 0.05
		self.warmup_steps = 10
		self.eval_per_epoch = 2
		self.log_result_step = 1


def seedSetting():
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)
	torch.backends.cudnn.deterministic = True  # 保证每次结果一样
def main():
	seedSetting()
	config = Config()
	train_dataset = ujson.load(open(config.train_path,'r'))
	dev_dataset = ujson.load(open(config.dev_path,'r'))
	test_dataset = ujson.load(open(config.test_path,'r'))
	train_iter = DatasetIterator(train_dataset, config.batch_size, shuffle=True,image_root_path=config.picture_path,with_label=True)
	dev_iter = DatasetIterator(dev_dataset, config.batch_size, shuffle=True,image_root_path=config.picture_path,with_label=True)
	test_iter = DatasetIterator(test_dataset, config.batch_size, shuffle=True,image_root_path=config.picture_path,with_label=True)
	model = vgg16_bn(config)
	trainer = Trainer(config,model)
	trainer.run_train(train_iter,dev_iter,test_iter)

if __name__ == "__main__":
	main()
	
	