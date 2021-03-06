# Cross-Domain Weakly-Supervised Segmentation test

This repository tests the method of [paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.html) on 
the semantic segmentation task. **(work in progress)**

This on-going work is planned as follows:
1. Crawl the naver-webtoon images
2. Domain transfer from PASCAL VOC to naver-webtoon images
3. Fine-tuning the segmentation model on the transferred PASCAL VOC2012 
4. Fine-tuning the segmentation model on the labeled naver-webtoon images **(current step)**


## Requirements
- Python 3.7+
- mmsegmentation
- mmcv-full
- pytorch
- torch-vision

You require a mmsegmentation to run the segmentation. Please follow the instruction of corresponding [repository](https://github.com/open-mmlab/mmsegmentation).

## Step 1. Crawl the naver-webtoon images
> We prepared the data with naver-webtoon-crawling repository [here](https://github.com/ckdghk77/naver-crawler).

> ![image](./figs/webtoon_example.png)


## Step 2. Domain transfer from PASCAL VOC2012 to naver-webtoon images
> We performed domain transfer with official pytorch Cycle-GAN implementation [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
> This code doesn't include the domain-transfer part. We assume that it dt_webtoon is prepared beforehand. You can simply get the same results if you train on
 the official Cycle-GAN code.

> ![image](./figs/dt_result_webtoon.png)

## Step 3. Fine-tuning the segmentation model on the transferred PASCAL VOC
> We performed finetuning with train.py. You can find the improved segmentation result below.
> We cannot conduct quantitative analysis since we don't have accessible annotation data.

> ![image](./figs/fine_tuned_result.png)


## Step 4. Fine-tuning the segmentation model on the labeled naver-webtoon images **(current step)**


