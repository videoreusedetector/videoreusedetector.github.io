# About

## Welcome to the Video Reuse Detector (VRD).

The VRD is a methodological toolkit for identifying visual similarities in audiovisual archives with the help of machine learning methods. It has been assembled because of the lack of open-source solutions for audiovisual copy detection and is meant to help archivists and humanistic scholars study video reuse. 

By automating the process of detecting how, when, and where video content reemerges within a given archive, the VRD allows you to match and compare a) one or several selected reference videos against a larger database, or b) all videos within a database against each other.

The toolkit has been developed within the research project [European History Reloaded: Curation and Appropriation of Digital Audiovisual Heritage](https://www.cadeah.eu/), funded by the JPI Cultural Heritage project, EU Horizon 2020 research and innovation programme. Its main developer is [Tomas Skotare](https://www.umu.se/en/staff/tomas-skotare/), with assistance from [Maria Eriksson](https://www.umu.se/en/staff/maria-c-eriksson/) and [Pelle Snickars](https://www.umu.se/en/staff/pelle-snickars/). 


## Scientific rationale 

Our work with the VRD is inspired by the [“visual turn”](https://academic.oup.com/dsh/article/35/1/194/5296356) in historic and digitally-oriented humanities research, where a growing number of scholars have recently shifted their attention towards working with visual and audiovisual sources (see for example [Arnold and Tilton](https://www.distantviewing.org/), [Wevers and Smits](https://academic.oup.com/dsh/article/35/1/194/5296356), [Manovich](https://mitpress.mit.edu/books/cultural-analytics), [Kee and Campeau](https://hdl.handle.net/2027/fulcrum.70795899c)). 

The VRD builds on and extends these research efforts, but it also differs from existing tools in one important way: while many audiovisual toolkits in the digital humanities involving machine learning are focused on automating the production of semantic metadata – and use machines to annotate _who_ or _what_ appears in images – our tool is solely focused on searching for visual similarities within a given dataset. 

In short, we are interested in mapping the “social life” or “cultural biographies” of individual video clips (to borrow from [Arjun Appadurai](https://www.cambridge.org/core/books/social-life-of-things/4F4D3929A501EC19CF413D36BDF8AB3A) and [Igor Kopytoff](https://www.cambridge.org/core/books/social-life-of-things/4F4D3929A501EC19CF413D36BDF8AB3A)). This, for example, involves exploring how the meaning of historic footage changes when it circulates and is recycled/cross-referenced in video productions through time. 

How, for instance, has a selected video – say, footage from the moon landing in 1969 – been reused in documentary films and news broadcasts throughout history? When and where has this footage played a role in extraterrestrial imaginaries, or potentially helped shape narratives concerning fundamentally different topics?

Answering such questions by analyzing large reference archives manually would be very time-consuming, creating incentives for the development of tools that can automatically reduce the workload. The VRD is such a tool and helps speed up the process of identifying reuse in large audiovisual databases. 

Aside from being used in academic settings, we hope the VRD can help archives and cultural heritage institutions find excessive copies in digitized collections and otherwise assist in finding patterns of similarity across audiovisual databases.


# Readme


## Availability 

The VRD is embedded in [Jupyter Notebooks](https://jupyter.org/) – an interactive computing environment and programming interface that allows users to run customized code through their web browsers. 

We chose to work with Jupyter Notebooks because it is especially good for guiding someone through a computational workflow and explaining the basic logic and ideas behind software. Jupyter notebooks can also easily be extended, modified, and transformed in line with the user’s preferences, making it a suitable starting point for developing a research project.

Jupyter Notebooks is free to use and download and there are several tutorials and guides that explain how the tool works. [This is a good place to start](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). 


## Technical overview

The VRD uses machine learning techniques (more specifically, [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), or CNN’s), combined with tools for performing similarity searches (more specifically, the [Faiss library](https://ai.facebook.com/tools/faiss/)) to detect copies in audiovisual archives. 

The VRD also assembles a series of tools for trimming and preparing datasets (including software for removing black borders and monochrome video frames), and filtering/visualizing matching results (such as introducing similarity thresholds, filtering based on sequential matches of frames, and visually viewing the final matching results). 

We decided to work with CNN’s and the Faiss library after doing multiple tests with more traditional tools for visual content recognition, including video fingerprinting with help of [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF) (Oriented FAST and rotated BRIEF) – a method for extracting and comparing visual features in images. The combination of CNN’s and Faiss quickly outperformed the ORB technology’s ways of identifying visual similarities in video content, however, both in terms of accuracy and processing speed.


## A brief introduction to convolutional neural nets (CNN’s)

Convolutional neural nets, or CNN’s, constitute a machine learning technique that is frequently used to extract and analyze visual content. You can read more about how CNNs work [here](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks) or [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53), but put in simple terms, the technology is inspired by a model of the connectivity patterns of neurons in the animal visual cortex.  

CNN’s are for example used in medical image analysis, image recommendation systems, and – importantly for our purposes here – to automatically search for and identify similarities in images or videos. 

While the detailed technical workings of individual CNNs may differ, the technology is broadly designed according to multiple layers of analysis and abstraction. Each layer in a CNN processes an input and produces an output, which is passed on to the next layer. 

For instance, one layer in a CNN may observe how pixels are spatially arranged and search for areas with a high contrast between nearby pixels (a good marker for what is visually unique in a particular image), while another layer might focus on reducing what information is stored about pixel contrasts (instructing the model to “forget” all areas in a picture with a lower pixel contrast than a given value, for example). In this way, the CNN produces a successively smaller and hopefully more precise “map” of the analyzed image. 



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")
  

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



```
Example of what a processed frame may look like at one layer of the analysis when a CNN is applied to analyze video content. Here, we can assume that the CNN was searching for pixel contrasts by transforming the color settings in the analyzed frame (Image courtesy: European  History Reloaded Team). 
```


Somewhere before the top layers of the analysis is reached, a CNN produces a compressed and final interpretation of the key visual characteristics of an image. It is then common for the remaining layers of a CNN to be trained/designed to annotate or classify the content in images by estimating what objects, animals, and human faces appear in the footage, for example. 


## The VRDs use of CNN’s

In order to use a CNN, the VRD begins its analysis by dividing each video in a designated dataset into still frames. Commonly, a digital video consists of 24-30 frames per second, and the VRD is pre-set to extract one such frame for every second of video. 

The VRD then applies a CNN to process the individual frames but stops when the final (and compressed) interpretation of an image has been produced. These key visual features are exported as-is and because of their highly abstracted nature, they are somewhat resilient to modifications in the source data. This means that an image can be recognized even if someone has adjusted its color, resolution, or composition, for example.

That the VRD stops before the top layers in the CNN have been reached, means that the tool does not analyze and annotate what things, people, or places appear in images. Instead, it uses the compressed visual features to find patterns of similarity across images. This shortens the image processing time and is especially customized for studying video reuse.

Currently, the four different convolutional neural networks that come with [Keras](https://keras.io/) – a so-called “wrapper” for CNN’s – can be used with the VRD: [ResNet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html), [Inception_3](https://cloud.google.com/tpu/docs/inception-v3-advanced), [VGG16](https://arxiv.org/abs/1505.06798), and [MobileNets](https://arxiv.org/abs/1704.04861). Each of these CNN’s apply their own method for extracting visual features from images but they all exhibit a layered convolutional structure. What also combines them, is that they have been trained and validated using the [ImageNet dataset](http://www.image-net.org/) (or more precisely, the so-called [ImageNet Large Scale Visual Recognition Challenge](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)) which is arguably the world’s most frequently used dataset for developing and testing machine vision tools. 

[ResNet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html) and [VGG-16](https://arxiv.org/abs/1505.06798) have both been developed at Microsoft Research, while [Inception_V3](https://cloud.google.com/tpu/docs/inception-v3-advanced) and [MobileNets](https://arxiv.org/abs/1704.04861) have been developed at Google. It is difficult to say which one of these neural nets is “better” in the study of video reuse and they all come with individual [benefits and drawbacks](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/). 

Previous tests have shown that the performance and accuracy of the tools are [comparatively similar](https://keras.io/api/applications/), apart from the fact that the VGG-16 neural net is fairly large and uses quite a lot of disk space and bandwidth. It should be noted, however, that most performance tests of the above mentioned CNN’s have focused on their capacity to correctly annotate/classify the content in images (faces, animals, objects, etc.), and not their ability to assist in the detection of visual similarities, which is how the CNN’s are used in the VRD.

In our experiments with the VRD, we have mostly relied on ResNet50, although we encourage users to experiment and examine which CNN works best for their individual video dataset. While the current version of the VRD uses the CNNs that are included in Keras, there is also no reason why additional networks could not be applied in the future, potentially further improving performance. 


## Similarity search using Faiss

When the image processing with a CNN is finished, the VRD uses the so-called [Faiss](https://ai.facebook.com/tools/faiss/) (Facebook AI Similarity Search) library to calculate image similarity. Faiss was first released in 2019 and as the name suggests, it is developed by Facebook. Currently, it is considered to be one of the most efficient open-source tools for conducting large-scale similarity searches. For instance, Faiss can be run on Graphical Processing Units (GPU’s) which provides significant advantages in terms of speed. While Faiss can be used for any sort of similarity search, we use it here to identify similarities between images.

Faiss uses the visual features that were extracted in the previous step to index and calculate the visually most similar “neighbors” for each video frame. In more detail, all reference images are first added to a Faiss index. New images are then compared with all images in the index, producing an arbitrarily long (as defined by the user) list of similarity neighbours. Requesting more neighbours requires more processing time, meaning that the number of neighbours that the VRD is instructed to find is a tradeoff between time and accuracy. 

The found neighbours are each sorted by distance metric from the compared image, with lower numbers representing a more similar image. The distance metric 0.0 represents the absolute closest similarity match that the VRD can identify. What distance metric represents a “correct” match is hard to determine in a general sense, however, as it changes with the choice of a neural network, the quality of the source material, and the number of videos/images/frames in the index, for example.


## Filtering options

To tailor and narrowing down the search results, the beta version of the VRD includes a number of filtering options. For example, it is possible to filter out the display of any identified similarity neighbors that come from the same video. It is also possible to filter out monochrome frames from the search results, assuming that monochrome frames (such as totally black or white frames) are of less interest to the user. 

Users are also given the option to select a threshold for which distance metrics should be shown in the final matching results. For instance, the VRD may be instructed to only show frame pairs with an assigned distance metric of less than 30 000. If this threshold is accurate, it should greatly reduce the number of shown non-matching frames. 


## Matching results

When the VRD’s image processing and filtering is finished, the final matching results are shown to the user in textual and visual form. 

For instance, the user can study the matching results in tables that show which individual frames received the most identified similarity neighbours, or what frame pairs (i.e. comparisons between two different frames stemming from two different videos) received the lowest distance metrics.

[Insert image example]

Users can also explore visual samples of identified similarity matches between videos, by studying miniature images of the analyzed frames.

[Insert image example]

In addition, the VRD is equipped with possibilities to preview so-called sequential frame matches. We define a sequential match as an instance where two or more sequential frames (i.e. frames that were extracted one after the other from the original files) from two videos have been given a distance metric below a specified value. 

If frame 1-6 in Video X and frame 11-16 in Video Y are each given a distance metric below the threshold 20 000, for example, this may be defined as a sequential match. 

Sequential filtering can be used to identify instances when longer chunks of moving images have been reused,  and can for example be filtered according to the length of sequences (such as 3 seconds, 4 seconds or 5 seconds long sequential matches), or identified sequential matches for individual videos. 

[Insert image example]

On the whole, the above-mentioned lists, tables, and visual frame comparisons are meant to function as a guide that can point users towards videos that might be interesting to study in more detail – primarily by actually opening the original video files and viewing the moving images. 

In other words, we strongly advise against exporting and using these statistics as absolute proof of video reuse and instead encourage users to approach them as an _assistance tool_ in navigating large video datasets.


## Toolkit limitations

In its current version, the VRD only analyzes visual content and disregards sound. This means that the tool cannot detect if the audio track of a video file has been reused and reappears somewhere else. The rationale for this is that reused video is often dubbed/overlayed with different audio, meaning that the audio might not match even in cases where the video is reused. Since we were originally interested in performing visual similarity searches, adding audio features to the VRD has not been a priority at this stage. There is no reason why sound recognition technologies could not be incorporated in the tool in the future, however. 

By testing and analyzing the VRD’s matching results against the [VCDB dataset](http://www.yugangjiang.info/research/VCDB/index.html) (a large collection of remixed and manually annotated videos originating from video platforms such as YouTube), we have identified a few areas where the tool in its current state produces subpar results. One instance when the VRD’s performance generally suffers is when the source video is of very low quality, such as when it is heavily distorted/modified, has a low-resolution, or has heavy textual or symbolic overlays (such as subtitles, logotypes, and news show banners). 

Furthermore, the VRD has difficulties recognizing content if it is faced with the challenge of identifying content shown on a hand-filmed TV or computer screen (a problem that is generally known as picture-in-picture). Another source of concern has been video content with large amounts of text, since text can easily be matched against text in other videos, even if the precise use of words are not identical. 

Regardless of these limitations, however, there is still a good chance that the VRD will correctly identify even highly modified or overlaid content – a chance that can in many cases be further improved by pre-processing.

Finally, it is important to point out that the VRD produces an _estimation _of patterns of similarity across video files. This means that its outputs/results should not be interpreted as a definitive measurement of similarity or a final and absolute proof of video reuse. 



# Install


## Download Docker

The VRD is designed to be run and installed through Docker – a third-party application that enables the use of containerized applications. To begin with, you must therefore download and install the Docker software.

[Click here](https://docs.docker.com/get-docker/) to find the Docker installation packages and instructions for your operating system. 


## Install the VRD Docker container 

When Docker is installed, you can download and install the specialized VRD docker container by clicking on this link. 

This will create a docker image that contains all the necessary software packages required to run the VRD. 


## Adjust local configurations

Before the VRD is ready to be used, you need to edit the docker-compose.yml file to match your local file system. 

Open this file in your terminal and look for lines 14-15, which should say:

-'/{localpath}/VRD/Docker/ai-notebook/mount/Notebooks/:/home/jovyan/notebooks'

- '/{localpath}/VRD/Videos:/home/jovyan/Videos/'

Then change both {localpath} lines to correctly match the directory you downloaded to. 

_[insert example of what this might look like, which elements need to be changed]_

When this is done, open a terminal (or equivalent) and navigate to the VRD/Docker directory on your system, and type docker-compose up. 

This should start the creation of the docker container, although it may take some time and require a couple of gigabytes of data to be downloaded. 


## Access the VRD via Jupyter Labs

We suggest you view and use the VRD notebook via the web interface [Jupyter Labs](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906). 

When the previous steps in the installation guide have been finished, the docker container shown in your terminal should present you with a link to a homepage, similar to this:

http://127.0.0.1:9003/lab/tree/notebooks/AI%20Notebook%20-%20SVT%201909.ipynb

Copy and insert this link in your preferred web browser. You should now be able to access the VRD via Jupyter Labs and start working on your project.



# Contact

Follow our project on [GitHub](https://github.com/inidun) and the [European History Reloaded website](https://www.cadeah.eu/). 

Don’t hesitate to reach out to [tomas.skotare@umu.se](mailto:tomas.skotare@umu.se) or [maria.c.eriksson@umu.se](mailto:maria.c.eriksson@umu.se) if you have any questions or comments.
