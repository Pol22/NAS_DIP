import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='path to image for DIP',
                    required=True)
parser.add_argument('--scale', type=int, help='scale for SR image',
                    default=4)
parser.add_argument('--depth', type=int, help='depth of encoder-decoder',
                    default=4)
parser.add_argument('--epochs', type=int, help='num of training epochs',
                    default=2000)
parser.add_argument('--population', type=int, help='size of population',
                    default=20)
parser.add_argument('--generations', type=int, help='number of generations',
                    default=30)
parser.add_argument('--mutation_prob', type=float, help='mutation prob',
                    default=0.05)
parser.add_argument('--gpus', type=int, help='number of gpus',
                    default=8)
# TODO append train timeout
# TODO append flag for saving images for models
args = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import save_model
import subprocess
from threading import Thread, Lock
from queue import Queue
from time import time
from datetime import datetime
from PIL import Image
import numpy as np

import unit
import model_generation


TRAIN_SCRIPT = 'train.py'
LOG_FILE = 'train.log'
TMP_IMG = 'tmp.png'
TRAIN_DIR = './train_dir/'
MODEL_NAME_SUFFIX = '.h5'
IMG_SUFFIX = '.png'
INPUT_DEPTH = 32
MAX_PARAMS = 15 * 1e6 # 15 mil


def logging(string):
    with open(LOG_FILE, 'a') as log:
        log.write(string)
        log.write('\n')


class Trainer(Thread):
    '''
        Deamon thread for training
    '''
    # static variables
    metrics = list()
    models = list()
    lock = Lock()

    def __init__(self, queue, gpu_index):
        super(Trainer, self).__init__()
        self.queue = queue
        self.gpu_index = gpu_index

    def run(self):
        while True:
            model_path = self.queue.get()
            self.train(model_path)
            self.queue.task_done()

    @staticmethod
    def update(metric, model):
        # thread lock
        Trainer.lock.acquire()
        for i in range(len(Trainer.metrics)):
            if metric > Trainer.metrics[i]:
                Trainer.metrics.insert(i, metric)
                Trainer.models.insert(i, model)
                Trainer.lock.release()
                return

        Trainer.metrics.append(metric)
        Trainer.models.append(model)
        Trainer.lock.release()

    @staticmethod
    def clear():
        Trainer.metrics.clear()
        Trainer.models.clear()

    def train(self, model_path):
        print(f'Start training model: {model_path}')
        train_args = ['python', TRAIN_SCRIPT, '--model', model_path, '--img',
                      args.img, '--epochs', str(args.epochs),
                      '--gpu', str(self.gpu_index)]
        start = time()
        process = subprocess.run(train_args, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        end = time()
        last_line = process.stdout.decode().split()[-1]
        # TODO getting other metrics
        psnr = float(last_line.split('=')[-1])
        log_str = f'[{psnr:2.4f} / {(end-start)/60:2.2f} min] {model_path}'
        print(log_str)
        logging(log_str)
        # update top
        Trainer.update(psnr, model_path)


def model_params_count(model):
    result = 0
    for weight in model.trainable_variables:
        result += np.prod(weight.shape)
    return result


class MutationController(object):
    def __init__(self):
        # paired containers (always changes together)
        self.metrics = list()
        self.bin_models = list()
        self.models = list()

    def print_top(self):
        for i, metric in enumerate(self.metrics):
            log_str = f'[{metric:2.4f}] {self.models[i]}'
            print(log_str)
            logging(log_str)

    def insert(self, metrics, models):
        assert(len(metrics) == len(models))
        for i in range(len(metrics)):
            # sorted insert for each model
            for j in range(len(self.metrics) + 1):
                # lazily condition for last position
                if j == len(self.metrics) or metrics[i] > self.metrics[j]:
                    if models[i].endswith(MODEL_NAME_SUFFIX):
                        self.metrics.insert(j, metrics[i])
                        model_name = os.path.split(models[i])[-1]
                        hex_model = model_name.split(MODEL_NAME_SUFFIX)[0]
                        bin_model = unit.hex_to_bin_list(hex_model,
                                                         args.depth)
                        self.bin_models.insert(j, bin_model)
                        self.models.insert(j, models[i])
                        break

    def _mutation(self, bin_model):
        mutation_rand = np.random.uniform(size=len(bin_model))
        for i in range(len(bin_model)):
            if mutation_rand[i] < args.mutation_prob:
                bin_model[i] = not bin_model[i] # flip bit
        return bin_model

    def _select_parents(self):
        norm_metrics = self.metrics / np.sum(self.metrics) # normalization
        cum_sum = np.cumsum(norm_metrics)
        rand_num1 = np.random.uniform()
        rand_num2 = np.random.uniform()
        index1 = int(np.sum(cum_sum < rand_num1))
        index2 = int(np.sum(cum_sum < rand_num2))
        return self.bin_models[index1], self.bin_models[index2]

    def crossover(self):
        parent1, parent2 = self._select_parents()
        unit_length = 14 + 4 * args.depth
        split_unit = np.random.randint(0, args.depth)
        result = np.concatenate((parent1[:split_unit*unit_length],
                                 parent2[split_unit*unit_length:]))
        return self._mutation(result)

    def selection(self):
        # metric = np.mean(self.metrics)
        # log_str = f'\nMEAN: {metric}'

        # median is better then mean
        metric = self.metrics[len(self.metrics) // 2]
        log_str = f'\nMEDIAN: {metric}'
        logging(log_str)
        print(log_str)
        for _ in range(len(self.metrics)):
            if self.metrics[-1] < metric:
                os.remove(self.models[-1])
                os.remove(self.models[-1] + IMG_SUFFIX)
                self.metrics.pop()
                self.bin_models.pop()
                self.models.pop()
            else:
                return


def main():
    # create train dir
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    # remove previous train log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # create queue for models
    queue = Queue()
    # create training thread for each gpu
    for gpu_index in range(args.gpus):
        trainer = Trainer(queue, gpu_index)
        trainer.setDaemon(True)
        trainer.start()

    # read hr image to get img size
    img = Image.open(args.img)
    w, h = img.size
    # condition of encoder-decoder model
    if w % (2 ** args.depth) == 0 and h % (2 ** args.depth):
        w -= w % (2 ** args.depth)
        h -= h % (2 ** args.depth)

        crop = img.crop((0, 0, w, h))
        crop.save(TMP_IMG)
        args.img = TMP_IMG

    # MutationController
    controller = MutationController()

    for generation in range(args.generations):
        print(datetime.now())
        logging(str(datetime.now()))
        log_str = f'\nTrain generation {generation}'
        print(log_str)
        logging(log_str)

        for _ in range(args.population):
            while 1 > 0:
                # TODO check if rand generation using change increasing psnr 
                if generation == 0:
                    # init random models
                    bin_model = unit.gen_bin_model(args.depth)
                else:
                    # crossover 
                    bin_model = controller.crossover()
                units = unit.bin_model_to_units_list(bin_model, args.depth)
                model = model_generation.units_list_to_model(
                    units, (w, h, INPUT_DEPTH), args.depth)
                if model is not None:
                    params_count = model_params_count(model)
                    # limit number of parameters
                    if params_count > MAX_PARAMS:
                        continue
                    model_name = unit.bin_list_to_hex(bin_model, args.depth)
                    model_path = os.path.join(
                        TRAIN_DIR, model_name + MODEL_NAME_SUFFIX)
                    # check already trained
                    if os.path.exists(model_path):
                        continue
                    save_model(model, model_path, include_optimizer=False)
                    # put model on queue for training
                    queue.put(model_path)
                    break
        # wait training all models
        queue.join()
        # append trained models to MutationController
        controller.insert(Trainer.metrics, Trainer.models)
        # clear Trainer models
        Trainer.clear()
        # models selection
        controller.selection()
        # log
        log_str = f'\nTop of generation {generation}'
        print(log_str)
        logging(log_str)
        controller.print_top()

    print(datetime.now())
    logging(str(datetime.now()))


if __name__ == '__main__':
    main()
