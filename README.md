# Consensus Focus for Object Detection and minority classes

## Abstract
Ensemble methods exploit the availability of a given number of classifiers or detectors trained in single or multiple source domains and tasks to address machine learning problems such as domain adaptation or multi-source transfer learning. Existing research measures the domain distance between the sources and the target dataset, trains multiple networks on the same data with different samples per class, or combines predictions from models trained under varied hyperparameters and settings. Their solutions enhanced the performance on small or tail categories but hurt the rest. To this end, we propose a modified consensus focus for semi-supervised and long-tailed object detection. We introduce a voting system based on source confidence that spots the contribution of each model in a consensus, lets the user choose the relevance of each class in the target label space so that it relaxes minority bounding boxes suppression, and combines multiple models' results without discarding the poisonous networks. Our tests on synthetic driving datasets retrieved higher confidence and more accurate bounding boxes than the NMS, soft-NMS, and WBF.

**Keywords**: ensemble methods, object detection, consensus, long-tailed learning.

## Requirements

You can install the minimum requirements via pip or conda as follows:

**pip**
```
pip install -r requirements.txt
```

**Conda**
```
conda env create -f environment.yml 
conda activate Consensus_WBF1
```

You can also find all notebooks available in Google Colab. Please click on them, and you will find a link to see them.

## Description

### demo1.ipynb
This Jupyter notebook contains an example that uses offline inferences from three source networks to compute the consensus WBF. 

Please download the target dataset (17-00_CLEAR_SKY_yolov8) and unzip it in the main root:

```
wget https://yolov5defects.s3.us-west-1.amazonaws.com/17-00_CLEAR_SKY_yolov8.zip
unzip 17-00_CLEAR_SKY_yolov8.zip
rm -rf 17-00_CLEAR_SKY_yolov8.zip
```

## Implementation details
Our experiments were performed on a PC with an Intel(R) Core(TM) i5-10400F 2.90GHz CPU and an NVIDIA RTX 3060 GPU.

## Additional information about labels

We considered three synthetic datasets for autonomous driving: Apollo Synthetic, FCAV, and Virtual KITTY 2. Since the first has more than 273k distinct images, our tests take two out of seven parts (13-00 and 18-00) with clear and heavy rain, all degradations, pedestrians, traffic barriers, and all scenes. The target dataset belongs to an Apollo subset created by simulating the daytime at 5 pm with a clear sky and including the previous environmental variations. Each target class has 1333, 4556, and 234 samples, accordingly. For this experiment, we group all the categories into three classes to homogenize the label space: pedestrian, motorized vehicle (e.g., car, pickup, truck, etc.), and non-motorized-vehicle (cyclist, motorcyclist, unicyclist, etc.).

## YOLOv8 weights

YOLOv8x is the benchmark model to compare the performance of our approach with the NMS, soft-NMS, and WBF. You can dowload them using the following links:

```
wget https://yolov5defects.s3.us-west-1.amazonaws.com/13_CLEAR_SKY_best.pt
wget https://yolov5defects.s3.us-west-1.amazonaws.com/18_CLEAR_SKY_best.pt
wget https://yolov5defects.s3.us-west-1.amazonaws.com/fcav_best.pt
wget https://yolov5defects.s3.us-west-1.amazonaws.com/vkitti_best.pt
```

**You can obtain the ensembled labels through NMS, soft-NMS, and WBF by following the instructions in other_ensembles.ipynb**

## Citation

@misc{salgado2024consensus,
      title={Consensus Focus for Object Detection and minority classes}, 
      author={Erik Isai Valle Salgado and Chen Li and Yaqi Han and Linchao Shi and Xinghui Li},
      year={2024},
      eprint={2401.05530},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}