import numpy as np
from numpy import asarray
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
from .u2net import realesrgan
from .u2net import utils, model, tracer
from pathlib import Path
import os
from django.conf import settings
import onnx
import onnxruntime
from onnx2torch import convert
import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
from torchvision import transforms

import pyanime4k
from .u2net import config
import torch.nn as nn


# Upscale Model

model_upscale = realesrgan.RealESRGAN('cuda', scale=4)
model_upscale.load_weights('main/ckpt/RealESRGAN_x4plus_anime_6B.pth')



# Delete Background Model

model_name = 'TRACER-Efficient-7.pth'
model_path = os.path.join(settings.BASE_DIR, 'main/ckpt/' + model_name)
refiner = refine.Refiner(device='cuda')  # device can also be 'cpu'

torch.cuda.empty_cache()

if model_name == 'u2net.pth':
    model_pred = model.U2NET(3, 1)
    model_pred.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    model_pred.eval()
elif model_name == 'u2netp.pth':
    model_pred = model.U2NETP(3, 1)
    model_pred.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    model_pred.eval()
elif model_name == 'TRACER-Efficient-4.pth':
    cfg = config.getConfig()
    model_pred = tracer.TRACER(cfg).to('cuda:0')
    model_pred = nn.DataParallel(model_pred).to('cuda:0')
    model_pred.load_state_dict(torch.load(model_path, map_location="cuda:0"), strict=False)
    model_pred.eval()
else:
    model_pred = onnxruntime.InferenceSession(model_path)
    # model_pred = onnx.load(model_path)
    # # model_pred = convert(onnx_model)
    # # onnx_model = onnx.load(model_path)
    # # model_pred.load_state_dict(torch.load(ConvertModel(onnx_model), map_location="cpu"))
    # model_pred.eval()

def remove_transifer(image):
    model_pred = model.U2NET(3, 1)
    model_pred.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    model_pred.eval()
    sample = preprocess(np.array(image))
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float()).cuda()
        predictions = model(inputs_test)
        return predictions

def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    if model_name == 'u2net.pth':
        transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    elif model_name == 'TRACER':
        # TRACER
        transform = transforms.Compose([utils.RescaleT(448), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def remove_bg(image, resize=False):
    sample = preprocess(np.array(image))

    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float()).cuda()
        # print(sample["image"])
        if model_name == 'u2net.pth':
            d1, _, _, _, _, _, _ = model_pred(inputs_test)
        else:
            # print(torch.tensor(np.array(image), device='cuda:0', dtype=torch.float32))
            d1, _, _ = model_pred(inputs_test)
        torch.cuda.empty_cache()
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        img_out = img_out.resize((image.size), resample=Image.BILINEAR)
        empty_img = Image.new("RGBA", (image.size), 0)
        enhancer = ImageEnhance.Sharpness(img_out)
        factor = 5
        img_out = enhancer.enhance(factor)
        # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpen = cv2.filter2D(np.array(img_out), -1, sharpen_kernel)
        #
        # cv2.imshow('image', img_out.convert("L"))
        # cv2.waitKey()
        # print(Image.fromarray(sharpen * 255).resize(image.size))

        # out_1 = Image.fromarray(predict * 255).convert("RGB").filter(ImageFilter.GaussianBlur(2))
        # enhancer_object = ImageEnhance.Sharpness(out_1)
        # out = enhancer_object.enhance(5)

        # out = Image.fromarray(predict * 255).convert("RGB").filter(ImageFilter.GaussianBlur(2))
        # # out = out.filter(ImageFilter.EDGE_ENHANCE)
        # # out = out.filter(ImageFilter.EDGE_ENHANCE_MORE)
        # out = out.filter(ImageFilter.UnsharpMask(radius=50, percent=250, threshold=5))
        #
        # out = ImageEnhance.Contrast(out)
        # out = out.enhance(2)
        #
        # out.save('sr_image5.png')

        # upscale_mask = pyanime4k.upscale_images('sr_image5.png')
        # print(upscale_mask)

        # sr_image = model_upscale.predict(Image.fromarray(predict * 255).convert("RGB"))
        # sr_image.save('sr_image2.png')
        # print(sr_image)
        # testMask = cv2.morphologyEx(np.array(Image.fromarray(predict * 255).convert("RGB")), cv2.MORPH_OPEN,
        #                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # print(testMask)
        img_out = Image.composite(image, empty_img, Image.fromarray(predict * 255).resize(image.size).convert("L"))

        # torch.cuda.empty_cache()


        # Save Images
        imageName = "Test"
        maskName = "TestMask"

        #
        # data = asarray(image)
        # print(data.shape)
        #
        # print(transforms.ToPILImage()(np.uint8(image)))


        img = cv2.cvtColor((np.array(image)), cv2.COLOR_RGBA2BGR)
        # #
        # img2 = cv2.cvtColor((np.array(img_out)), cv2.COLOR_RGBA2BGR)
        # cv2.imwrite('out_1.png', img)
        # # out_1 = pyanime4k.upscale_images('out_1.png')
        # # #
        # output = refiner.refine(img, np.array(Image.fromarray(np.array(Image.fromarray(predict * 255).convert("RGB"))).resize(image.size).convert("L")), fast=False, L=600)
        # output = refiner.refine(img, np.array(Image.fromarray(predict * 255).resize(image.size).convert("L")),
        #                         fast=False, L=600)
        #
        # cv2.imwrite('sr_image5_out.png', output)
        # upscale_mask = pyanime4k.upscale_images('sr_image5_out.png')
        # # # Image.fromarray(output).save('output_1.png')
        # torch.cuda.empty_cache()
        # # #
        # # # output_2 = pyanime4k.upscale_images('output_1.png')
        torch.cuda.empty_cache()
        # # # print(pyanime4k.upscale_images(output))
        # #
        # # # #
        # Image.composite(model_upscale.predict(image), empty_img, Image.fromarray(output_2).resize(image.size).convert("L"))
        # # #
        # # # out = Image.fromarray(output).resize(image.size).filter(ImageFilter.GaussianBlur(1))
        # # # # out = out.filter(ImageFilter.EDGE_ENHANCE)
        # # # # out = out.filter(ImageFilter.EDGE_ENHANCE_MORE)
        # # # out = out.filter(ImageFilter.UnsharpMask(radius=50, percent=150, threshold=5))
        # img_out2 = Image.composite(image, empty_img, Image.fromarray(output).resize(image.size).convert("L"))

        # cv2.imshow('output', img_out)
        # cv2.waitKey()

        del d1, pred, predict, inputs_test, sample

        torch.cuda.memory_summary(device=None, abbreviated=False)

        return img_out

def _remove(image):
    sample = preprocess(np.array(image))
    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        image = image.resize((img_out.size), resample=Image.BILINEAR)
        empty_img = Image.new("RGBA", (image.size), 0)
        img_out = Image.composite(image, empty_img, img_out.convert("L"))

    return img_out


def remove_bg_mult(image):
    img_out = image.copy()
    for _ in range(4):
        img_out = _remove(img_out)

    img_out = img_out.resize((image.size), resample=Image.BILINEAR)
    empty_img = Image.new("RGBA", (image.size), 0)
    img_out = Image.composite(image, empty_img, img_out)
    return img_out


def change_background(image, background):
    background = background.resize((image.size), resample=Image.BILINEAR)
    img_out = Image.alpha_composite(background, image)
    return img_out