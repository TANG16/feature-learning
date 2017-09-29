# Usage example

## Build docker image

* docker build -t "featurelearning:\<tag\>" .

** (OBS) The built container will not be using the GPU (and training will be orders of magnitude slower). To enable GPU, append -gpu to tensorflow docker image (e.g. FROM tensorflow/tensorflow:0.7.1-gpu) in Dockerfile before building. When running, use nvidia-docker instead of docker. Though, currently the minimum required Cuda capability is 3.5. To enable lower Cuda capability, tensorflow has to be built from source

## Learn features useful for e.g. image similarity

* docker run -t -v \<host_image_folder\>:/images featurelearning:\<tag\> python vionel_feature_learning/examples/train_similar_images.py -images_directory /images

## Illustrations

### Image similarity examples (query image upper left - closest images upper row - furthest images lower row)

![Alt text](/readme_images/similar_images_1.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_2.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_3.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_4.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_5.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_5.png?raw=true "Optional Title")

![Alt text](/readme_images/similar_images_5.png?raw=true "Optional Title")

### Moodboard examples (images with the most of a given learnt feature/mood/style; one feature/mood/style per image)

![Alt text](/readme_images/moodboard_1.png?raw=true "Optional Title")

![Alt text](/readme_images/moodboard_2.png?raw=true "Optional Title")

![Alt text](/readme_images/moodboard_3.png?raw=true "Optional Title")

![Alt text](/readme_images/moodboard_4.png?raw=true "Optional Title")

### CNN autoencoder illustrations

![Alt text](/readme_images/conv_autoencoder.png?raw=true "Optional Title")

![Alt text](/readme_images/cnn_features.png?raw=true "Optional Title")

![Alt text](/readme_images/faces_cars_elephants.png?raw=true "Optional Title")
