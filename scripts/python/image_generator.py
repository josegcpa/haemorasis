"""
Functions and classes for tile generators from histopathology slides
"""

import numpy as np
import openslide
from openslide import OpenSlideError
from multiprocessing import Queue,Process

def image_generator(quality_csv_path,slide_path,
                    h=512,w=512,extra_padding=128):
    OS = openslide.OpenSlide(slide_path)
    dims = OS.dimensions
    with open(quality_csv_path) as o:
        lines = [x.strip() for x in o.readlines() if 'OUT,' in x]
        positives = []
        negatives = []
        for line in lines:
            data = line.split(',')
            if float(data[-1]) >= 0.5:
                positives.append([int(data[1]),int(data[2])])
            else:
                negatives.append([int(data[1]),int(data[2])])
    
    for x,y in positives:
        try:
            x = x - extra_padding
            y = y - extra_padding
            x = np.maximum(0,x)
            y = np.maximum(0,y)
            if x + h + (2*extra_padding) > dims[0]:
                x = dims[0] - h - (extra_padding*2)
            if y + w + (2*extra_padding) > dims[1]:
                y = dims[1] - w - (extra_padding*2)
            image = OS.read_region(
                (x,y),0,
                (h+(extra_padding*2),w+(extra_padding*2)))
            image = np.array(image)[:,:,:3]
            yield image,[x,y]
        except OpenSlideError as error:
            OS = openslide.OpenSlide(slide_path)

def image_generator_slide(slide_path,
                          height=512,width=512):
    OS = openslide.OpenSlide(slide_path)
    dim = OS.dimensions
    for x in range(0,dim[0],height):
        for y in range(0,dim[1],width):
            try:
                im = OS.read_region((x,y),0,(height,width))
                im = np.array(im)
                im = im[:,:,:3]
                yield im,'{},{}'.format(x,y)
            except OpenSlideError as error:
                OS = openslide.OpenSlide(slide_path)

class ImageGeneratorWithQueue:
    def __init__(self,slide_path,csv_path,
                 extra_padding=128,maxsize=1,
                 height=512,width=512):
        self.maxsize = maxsize
        self.csv_path = csv_path
        self.slide_path = slide_path
        self.extra_padding = extra_padding
        self.height = height
        self.width = width

        self.q = Queue(self.maxsize)
        self.p = Process(
            target=self.image_generator_w_q,
            args=(self.q,csv_path,slide_path,extra_padding))

    def image_generator_w_q(self,q,csv_path,slide_path,extra_padding):
        if csv_path != None:
            im_gen = image_generator(
                quality_csv_path=csv_path,
                slide_path=slide_path,
                extra_padding=extra_padding)
        else:
            im_gen = image_generator_slide(
                slide_path,
                self.height,self.width)
        for element in im_gen:
            q.put(element)
        q.put(None)

    def start(self):
        self.daemon = True
        self.p.start()

    def generate(self):
        while True:
            item = self.q.get()
            if item is not None:
                yield item
            else:
                self.p.join(120)
                break