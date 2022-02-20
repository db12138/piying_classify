import torch 
from torchvision import transforms
import random
from PIL import Image


class BasicDatasetIterator(object):
	def __init__(self, batches, batch_size, shuffle):
		if shuffle:
			random.shuffle(batches)
		self.batches = batches

		self.batch_size = batch_size
		self.n_batchs = int(len(batches) / self.batch_size)
		self.cur_batch = 0
		self.residue = len(batches) % self.batch_size

	def __iter__(self):
		return self

	def __len__(self):
		return self.n_batchs + int(not self.residue == 0)

	def creat_model_input(self, batch):
		return batch

	def _to_tensor(self, batch):
		return batch

	def __next__(self):
		if self.cur_batch < self.n_batchs:
			start_index = self.cur_batch * self.batch_size
			end_index = start_index + self.batch_size
			self.cur_batch += 1
			batch = self.batches[start_index:end_index]
			inputs = self.creat_model_input(batch)
			tensors = self._to_tensor(inputs)

			# if self.cur_batch == 1:
			# 	print("*****example*******")
			# 	print(inputs)
			return tensors

		elif self.cur_batch == self.n_batchs and self.residue:
			start_index = self.cur_batch * self.batch_size
			self.cur_batch += 1
			batch = self.batches[start_index:]
			inputs = self.creat_model_input(batch)
			return self._to_tensor(inputs)
		else:
			self.cur_batch = 0
			raise StopIteration

class DatasetIterator(BasicDatasetIterator):
	def __init__(self, batches, batch_size, shuffle,image_root_path,with_label=True):
		BasicDatasetIterator.__init__(self, batches, batch_size, shuffle)
		self.image_root_path = image_root_path
		self.with_label = with_label
		self.convert_tensor = transforms.ToTensor()
	def creat_model_input(self, batch):
		inputs = []
		labels = []
		for img_file_name,label in batch:
			cur_img_path = self.image_root_path + "/" +str(label)+ "/" + img_file_name
			img = Image.open(cur_img_path)
			cur_input = self.convert_tensor(img)
			cur_input = cur_input.unsqueeze(0)
			inputs.append(cur_input)
			labels.append(label - 1)
		inputs = torch.cat(inputs,dim=0)
		cur_batch = {"img_inputs":inputs,"labels":labels}
		return cur_batch
	def _to_tensor(self, batch):
		cur_batch_tensor= {}
		cur_batch_tensor["img_inputs"] = torch.tensor(batch["img_inputs"], dtype=torch.float)
		cur_batch_tensor['labels'] = torch.tensor(batch["labels"], dtype=torch.long)
		return cur_batch_tensor



