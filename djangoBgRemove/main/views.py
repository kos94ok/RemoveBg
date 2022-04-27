from django.shortcuts import render
from django.http import HttpResponse
from .forms import TaskForm
from PIL import Image
from io import StringIO
from django.core.files.storage import default_storage
from . import controllers
# from . import removeController
import uuid
import base64
import cv2
import numpy as np

import os
from django.conf import settings
import bgRemoverApp

model_name = "U2NET"  # U2NET, TRACER
model_path = "./main/ckpt/u2net.pth"
model_pred = bgRemoverApp.load_model(model_name, model_path)

# Create your views here.

def index(request):
    return render(request, 'main/index.html')

def postPage(request):

    # print(request.FILES['image'].read())
    #im = Image.open(StringIO(request.FILES['im'].read()))
    # im = Image.open(StringIO(request.FILES['image'].read()))
    #  Saving POST'ed file to storage

    savedInputImage = saveImage(request)
    # print(Image)

    input_image = Image.open(savedInputImage)
    # imageWithOutBg = removeController.remove_bg(input_image)
    imageWithOutBg = bgRemoverApp.remove_bg(input_image, model_name, model_pred, refine=False)
    imageName = uuid.uuid4()
    base64Image = proccessingImage(imageName, imageWithOutBg)

    # im_bytes = base64.b64decode(base64Image)
    # im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    # image = cv2.imdecode(im_arr, flags=cv2.IMREAD_UNCHANGED)

    # bgr = image[:, :, :3]  # Channels 0..2
    # alpha = image[:, :, 3]  # Channel 3
    # result = np.dstack([bgr, alpha])  # Add the alpha channel
    # cv2.imshow('Sharpen', result)
    # cv2.waitKey()
    # sharpen_kernel = np.array([[0, -1, 0],
    #                          [-1, 5, -1],
    #                          [0, -1, 0]])
    #
    # sharpen = cv2.filter2D(image, -1, sharpen_kernel)

    return HttpResponse(base64Image)


def saveImage(request):
    file = request.FILES['image']
    file_name = default_storage.save(file.name, file)
    #  Reading file from storage
    file = default_storage.open(file_name)
    file_url = default_storage.url(file_name)

    # Compress
    imageTemproary = Image.open(file)
    imageTemproaryName = 'media/result_' + file_name
    imageTemproary.save(imageTemproaryName, format='PNG', quality=60)

    return imageTemproaryName

def proccessingImage(imageName, imageWithOutBg):
    output_image = os.path.join(settings.BASE_DIR, 'media\\output\\' + f'{imageName}.png')
    imageWithOutBg.save(output_image)

    with open(output_image, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    return image_data

def cv2ToBase64(img):
    _, im_arr = cv2.imencode('.png', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64