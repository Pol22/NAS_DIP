import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model',
                    required=True)
parser.add_argument('--img', type=str, help='path to image for DIP',
                    required=True)
parser.add_argument('--scale', type=int, help='scale for SR image',
                    default=4)
parser.add_argument('--epochs', type=int, help='num of training epochs',
                    default=2000)
parser.add_argument('--gpu', type=int, help='on which gpu will be compute',
                    default=0) 
parser.add_argument('--timeout', type=int, help='training timeout (min)',
                    default=10)# TODO 20)
train_args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(train_args.gpu)

import tensorflow as tf
import numpy as np
from PIL import Image
from time import time


NOISE_STD = 1.0 / 10.0
REG_NOISE_STD = 0.05 # for scale 4


# class UpscaleLoss(tf.keras.losses.Loss):
#     def call(self, y_true, y_pred):
#         _, h, w, _ = y_pred.shape
#         downsampled = tf.image.resize(
#             y_pred,
#             size=[h // train_args.scale, w // train_args.scale],
#             method=tf.image.ResizeMethod.BILINEAR,
#             antialias=True)
#         diff = tf.math.squared_difference(y_true, downsampled)
#         return tf.reduce_mean(diff)


def train(model_path, img_path):
    hr_img = Image.open(img_path)
    lr_size = [hr_img.size[0] // train_args.scale,
               hr_img.size[1] // train_args.scale]
    lr_img = hr_img.resize(lr_size, Image.ANTIALIAS)
    model = tf.keras.models.load_model(model_path)
    # check SR compatibility
    # TODO crop output size image
    assert(model.outputs[0].shape[1] == hr_img.size[0])
    assert(model.outputs[0].shape[2] == hr_img.size[1])
    assert(model.outputs[0].shape[3] == 3)
    # TODO checks for other tasks
    # set input noise
    input_shape = list(model.inputs[0].shape)
    input_shape[0] = 1
    # const_noise = np.random.uniform(size=input_shape) * NOISE_STD
    const_noise = tf.random.uniform(shape=input_shape, maxval=1.0)
    const_noise *= NOISE_STD 

    lr_np = np.asarray(lr_img, dtype=np.float32)
    lr_np = lr_np / 255.0 # [0, 1]
    lr_np = np.expand_dims(lr_np, axis=0) # batch
    lr_np = np.transpose(lr_np, (0, 2, 1, 3))

    hr_np = np.asarray(hr_img, dtype=np.float32)
    hr_np = hr_np / 255.0 # [0, 1]
    hr_np = np.expand_dims(hr_np, axis=0) # batch
    hr_np = np.transpose(hr_np, (0, 2, 1, 3))

    optimizer = tf.keras.optimizers.Adam(1e-3)
    # loss_fn = UpscaleLoss()
    # TODO losses for other tasks
    mse = tf.keras.losses.MeanSquaredError()

    # start training (timeout)
    start = time()
    for epoch in range(train_args.epochs + 1):
        with tf.GradientTape() as tape:
            input_noise = const_noise + \
                tf.random.normal(shape=input_shape) * REG_NOISE_STD
            out_hr = model(input_noise, training=True)
            out_lr = tf.image.resize(out_hr, size=lr_size,
                method=tf.image.ResizeMethod.BILINEAR, antialias=True)
            loss = mse(lr_np, out_lr)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 20 == 0:
            # metric PSNR
            # psnr = tf.image.psnr(out_hr, hr_np, max_val=1.0).numpy()
            im1 = np.uint8(out_hr * 255.0)
            im2 = np.uint8(hr_np * 255.0)
            psnr = tf.image.psnr(im1, im2, max_val=255).numpy()
            # img1 = Image.fromarray(np.squeeze(im1))
            # img1.save('img' + str(epoch) + '.png')
            # img1 = Image.fromarray(np.squeeze(im2))
            # img1.save('img' + str(epoch) + '_.png')
            print(f'Epoch [{epoch:4d}/{train_args.epochs}]: ' + \
                  f'Loss: {float(loss):.5f}, PSNR: {psnr[0]:2.4f}')

            # timeout
            cur_time = time()
            elapsed_time = (cur_time - start) / 60 # sec / 60 = min
            if elapsed_time > train_args.timeout:
                break

    # save model
    tf.keras.models.save_model(model, model_path, include_optimizer=False)
    # save img
    im1 = np.transpose(np.squeeze(im1), (1, 0, 2))
    sr_img = Image.fromarray(im1)
    sr_img.save(train_args.model + '.png')

    return psnr[0]
    # return image
    # return model.predict(noise)


def main():
    result = train(train_args.model, train_args.img)
    # TODO create other evaluations (SSIM, LPIPS)
    # result = result * 255.0
    # img = Image.open(train_args.eval_img)
    # im1 = np.asarray(img, dtype=np.uint8)
    # im2 = np.uint8(np.squeeze(result))
    # sr_img = Image.fromarray(im2)
    # sr_img.save(train_args.model + '.png')
    # psnr = tf.image.psnr(im1, im2, max_val=255.0)
    print(f'PSNR={result}')


if __name__ == '__main__':
    main()
