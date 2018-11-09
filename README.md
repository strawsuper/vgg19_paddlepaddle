# VGG19 model for PaddlePaddle
The model is converted from Caffe model [VGG_ILSVRC_19_layers_deploy.prototxt](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt) and [VGG_ILSVRC_19_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel), using [caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid).

1.Download model and param
	* [Dropbox](https://www.dropbox.com/s/4rbkipqj2h86id6/VGG19_pd_model_param.tar.7z?dl=0)
2.Usage
- You can use like the [infer.py](./infer.py)
- You can also use directly
```
program, feed_names, fetch_targets = fluid.io.load_inference_model('./VGG19_pd_model_param', exe, 'vgg19_model', 'vgg19_params')
```