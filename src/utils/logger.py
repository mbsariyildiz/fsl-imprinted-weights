import os
import json
import numpy as np
import datetime
from collections import OrderedDict
from .misc import AverageMeter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorboard_logger import Logger as TBLogger

class Logger(object):

	def __init__(self, log_dir, label, titles, append_steps=1):
		"""
		log_dir      : str, directory where all the logs will be written.
		label        : str, root filename for the logs. It shouldn't contain an extension, such as .txt
		titles       : list, title for each log attribute.
		append_steps : int, 
		"""

		self.log_dir = log_dir
		self.label = label
		self.titles = titles
		self.append_steps = append_steps

		self.logs = {} # all title-log pairs that will be traced for this instance
		self.meters = {}
		for t in titles:
			self.logs[t] = []
			self.meters[t] = AverageMeter()

		if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
		self.tb_logger = TBLogger(self.log_dir)
		self.f_txt = open(os.path.join(self.log_dir, '{}.txt'.format(self.label)), 'w')

	def flush(self):
		self.save_as_arrays()
		self.save_as_figures()

	def close(self):
		self.flush()
		self.f_txt.close()

	def update(self, values, step):
		"""
		Adds a new log value for each title, also updates corresponding average meters.
		If step is multiple of append_steps, then self.append is called.

		values : list, must be of the same size as self.titles.
		step   : int, a step number
		"""
		assert len(self.titles) == len(values)

		for t, v in zip(self.titles, values):
			self.meters[t].update(v, 1)

		if step % self.append_steps == 0:
			values = [m.avg for m in self.meters.values()]
			self.append(values, step)

	def append(self, values, step):
		"""
		Adds a new log value for each title.

		values : list, must be of the same size as self.titles.
		step   : int, a step number
		"""
		assert len(self.titles) == len(values)
		
		step_log = OrderedDict()
		step_log['step'] = str(step)
		step_log['time'] = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

		for t, v in zip(self.titles, values):
			self.logs[t].append(v)
			step_log[t] = v
			self.tb_logger.log_value(t, v, step)

		json.dump(step_log, self.f_txt, indent=4)
		self.f_txt.write('\n')
		self.f_txt.flush()

	def save_as_arrays(self):
		"""
		Converts all logs to numpy arrays and saves them into self.log_dir.
		"""
		arrays = {}
		for t, v in self.logs.items():
			v = np.array(v)
			arrays[t] = v

		np.savez(os.path.join(self.log_dir, '{}.npz'.format(self.label)), **arrays)

	def save_as_figures(self):
		"""
		First, converts all logs to numpy arrays, then plots them using matplotlib. Finally, saves the plots into self.log_dir.
		"""
		for t, v in self.logs.items():
			v = np.array(v)

			fig = plt.figure(dpi=400)
			ax = fig.add_subplot(111)
			ax.plot(v)
			ax.set_title(t)
			ax.grid(True)
			fig.savefig(
				os.path.join(self.log_dir, '{}_{}.png'.format(self.label, t)),
				bbox_inches='tight' )
			plt.close()
		

