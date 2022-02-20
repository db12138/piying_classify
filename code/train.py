import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import logging
logger = logging.getLogger("__main__")

def accurate_num(logits, labels):
	_,predict_labels = torch.max(logits,1)
	predict_labels = predict_labels.view(-1)
	return torch.sum(predict_labels == labels).item()

def get_optimizer(model: nn.Module, args ) -> torch.optim.Optimizer:
	#no_decay = ['bias', 'LayerNorm.weight']
	# no_decay = []
	# optimizer_grouped_parameters = [
	# 	{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
	# 	 'weight_decay': weight_decay},
	# 	{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	# ]
	# optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

	optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
								momentum=args.momentum,
								weight_decay=args.weight_decay)


	return optimizer

def get_schedule_linear(optimizer, warmup_steps, training_steps, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases linearly after
	linearly increasing during a warmup period.
	"""
	def lr_lambda(current_step):
		if current_step < warmup_steps:
			return float(current_step) / float(max(1, warmup_steps))
		return max(
			0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer():
	def __init__(self,args,model):
		self.args = args
		self.model = model
		self.step = 0
		self.best_dev_acc = 0
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = get_optimizer(self.model,args)

	def run_train(self,train_iter,dev_iter,test_iter):
		updates_per_epoch = train_iter.__len__()
		self.model = torch.nn.DataParallel(self.model, self.args.device_ids)
		self.model.to(self.args.device_ids[0])

		total_updates = updates_per_epoch * self.args.epochs
		scheduler = get_schedule_linear(self.optimizer, self.args.warmup_steps, total_updates)
		eval_step = int(updates_per_epoch / self.args.eval_per_epoch)
		print(f"eval_step:{eval_step}")
		for epoch in range(0, int(self.args.epochs)):
			logger.info("***** Epoch %d *****", epoch)
			self._train_epoch(scheduler, epoch, eval_step, train_iter, dev_iter,test_iter)
	def _train_epoch(self,scheduler, epoch, eval_step, train_iter, dev_iter,test_iter):
		args = self.args
		self.model.train()
		self.model.zero_grad()
		step_loss = 0
		epoch_loss = 0
		for i,cur_batch in enumerate(tqdm(train_iter)):
			# for k,v in cur_batch.items():
			#     cur_batch[k] = v.to(f"cuda:{self.model.device_ids[0]}")
			inputs = cur_batch["img_inputs"]
			labels = cur_batch["labels"]

			outputs = self.model(inputs)
			
			loss = self.criterion(outputs.cpu(),labels)
			# print(f"outputs:{outputs.shape} , labels: {labels.shape}")
			# print(f"loss :{loss.shape}")
			loss = loss.mean()
			
			logits = outputs.cpu()
			# print(loss)
			loss.backward()
			# for name,p in self.model.named_parameters():
			# 	# print(name,p.requires_grad)
			# 	if name == "module.features.15.weight":
			# 		print(p)
			# 		# print(f"grad:{p.grad}")
			# 		print(f"require:{p.requires_grad}")
			# assert 0
			self.optimizer.step()

			self.optimizer.zero_grad()
			self.model.zero_grad()
			scheduler.step()
			step_loss += loss.item()
			if (i + 1) % args.log_result_step == 0:
				lr = self.optimizer.param_groups[0]['lr']
				average_loss = step_loss / args.log_result_step
				#logger.info('Epoch: %d: Step: %d, average_loss=%f, lr=%f', epoch, self.step, average_loss, lr)
				logger.info(f'Epoch: {epoch} Step: {self.step}, average_loss={average_loss}, lr={lr}')
				step_loss = 0

			if (i + 1) % eval_step == 0:
				logger.info('Validation: Epoch: %d Step: %d', epoch, self.step)
				# self.validate_and_save_Binary_Classify(epoch, train_iterator,dev_iterator)
				self.validate_and_save(epoch,dev_iter,test_iter)
				self.model.train()

	def validate_and_save(self,epoch,dev_iter,test_iter):
		self.model.eval()
		dev_right_num = 0
		dev_all_num = 0
		test_right_num  = 0
		test_all_num  = 0

		train_right_num = 0
		train_all_num = 0
		with torch.no_grad():
			for cur_batch in tqdm(dev_iter):
				inputs = cur_batch["img_inputs"]
				labels = cur_batch["labels"]
				outputs = self.model(inputs)

				loss = self.criterion(outputs.cpu(),labels)
				loss = loss.mean()
				logits = outputs.cpu()
				acc_num = accurate_num(logits, cur_batch['labels'])
				dev_right_num += acc_num
				dev_all_num += len(cur_batch['labels'])
			for cur_batch in tqdm(test_iter):
				inputs = cur_batch["img_inputs"]
				labels = cur_batch["labels"]
				outputs = self.model(inputs)

				loss = self.criterion(outputs.cpu(),labels)
				loss = loss.mean()
				logits = outputs.cpu()
				acc_num = accurate_num(logits, cur_batch['labels'])

				test_right_num += acc_num
				test_all_num += len(cur_batch['labels'])
		dev_acc = dev_right_num / dev_all_num
		test_acc  =test_right_num / test_all_num
		logger.info(f"epoch:{epoch},dev_acc:{dev_acc},test_acc:{test_acc}")

		if dev_acc > self.best_dev_acc :
			self.best_dev_acc = dev_acc
			torch.save({"net": self.model.state_dict(), "epoch": epoch, "dev_acc": dev_acc},
					   f"{self.args.task_name}_batch{self.args.batch_size}_lr{self.args.learning_rate}_epoch_{epoch}in{self.args.epochs}_{dev_acc}_{test_acc}.pt")


