# VGG19 model for PaddlePaddle
The model is converted from Caffe model [VGG_ILSVRC_19_layers_deploy.prototxt](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt) and [VGG_ILSVRC_19_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel), using [caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/caffe2fluid).

*If you want to use the whole model for the program, you can use load_inference_model to load the model.
1.Download model and params.
- [Dropbox for model and params](https://www.dropbox.com/s/4rbkipqj2h86id6/VGG19_pd_model_param.tar.7z?dl=0)

- Extract the model and params from the link, and the folder structure should looks like:
```
.
├── vgg19_model
├── vgg19_params
```

2.Usage

 - Extract the tar in the inference_model/ dir
- You can use directly:
```
program, feed_names, fetch_targets = fluid.io.load_inference_model('./VGG19_pd_model_param', 
					exe, 'vgg19_model', 'vgg19_params')
```
- You can see more detail in [inference_model/infer.py](./inference_model/infer.py)
####Advanced
If you want to just load the param and realize your own model to use and add the loss like me:
 
 - example:

(when I reimplement the SRGAN, the loss of Generator is three parts,`
g_loss = mse_loss + vgg_loss + g_gan_loss`

I need the vgg_api in the g_program.:+1:
So you can just use the load param.
```
fluid.io.load_params(exe, "./vgg19_pd_params")
```
 - See more detail in [load_params](load_params/infer.py)
 
  * You can download the params in different files [link for params](https://www.dropbox.com/s/7sae9pqf042llqq/vgg19_pd_params.tar.7z?dl=0)
  
  
  
  Thanks for oraoto!
I refer his code which is converted the [vgg16](https://github.com/oraoto/paddle-vgg16) for paddlepaddle.
  
  