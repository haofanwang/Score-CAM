## Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
To appear at IEEE [CVPR 2020 Workshop on Fair, Data Efficient and Trusted Computer Vision](https://fadetrcv.github.io).

<img src="https://github.com/haofanwang/Score-CAM/blob/master/pics/comparison.png" width="100%" height="100%">

In this paper, we develop a novel post-hoc visual explanation method called Score-CAM based on class activation mapping. Score-CAM is a gradient-free visualization method, extended from Grad-CAM and Grad-CAM++. It achieves better visual performance and fairness for interpreting the decision making process. 

Paper: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf) (Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel and Xia Hu.)

## Update
**`2020.5.11`**: Score-CAM has been merged into [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations).

**`2020.4.13`**: First version of Score-CAM code has been released. More implementations will be added later.

## Milestone
* - [ ] Support for Colab notebook.
* - [ ] Support for faster version of Score-CAM.
* - [ ] Support for pre-trained model in Pytorch.
* - [ ] Support for self-defined model in Pytorch.
* - [ ] Add visualization result and quantitive evaluation.
* - [ ] Support for other tasks such as object localization task.

## Implement Score-CAM into popular visualization tools.
It would be very appreciated for implementing Score-CAM for other popular projects, if any of you are interested.
* - [x] [issues #76](https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/76), Implement in [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* - [ ] [isses #350](https://github.com/pytorch/captum/issues/350), Implement in [pytorch/captum](https://github.com/pytorch/captum)
* - [ ] [issues #124](https://github.com/sicara/tf-explain/issues/124), Implement in [sicara/tf-explain](https://github.com/sicara/tf-explain)
* - [ ] Implement in [PAIR-code/saliency](https://github.com/PAIR-code/saliency)
* - [ ] Implement in [experiencor/deep-viz-keras](https://github.com/experiencor/deep-viz-keras)

## Other implementations
Before we release the official code, some great researchers have implemented Score-CAM on different framework. 
I am very grateful for the efforts made in their implementation.

### Pytorch:

[torch-cam](https://github.com/frgfm/torch-cam) by [frgfm](https://github.com/frgfm)

[ScoreCAM](https://github.com/yiskw713/ScoreCAM) by [yiskw713](https://github.com/yiskw713)

[xdeep](https://github.com/datamllab/xdeep) by [datamllab](https://github.com/datamllab)

### Tensorflow:

[score-cam](https://github.com/matheushent/score-cam) by [matheushent](https://github.com/matheushent)

### Keras:

[scam-net](https://github.com/andreysorokin/scam-net) by [andreysorokin](https://github.com/andreysorokin)

[Score-CAM](https://github.com/tabayashi0117/Score-CAM) by [tabayashi0117](https://github.com/tabayashi0117)

[Score-CAM-VGG16](https://github.com/bunbunjp/Score-CAM-VGG16) by [bunbunjp](https://github.com/bunbunjp)

## Blog post

[paper_summary](https://github.com/yiskw713/paper_summary/issues/98)

[Demystifying Convolutional Neural Networks using ScoreCam](https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-scorecam-344a0456c48e)

[kerasでScore-CAM実装．Grad-CAMとの比較](https://qiita.com/futakuchi0117/items/95c518254185ec5ea485)

## Citation
If you find this work or code is helpful in your research, please cite and star:
```
@InProceedings{Wang_2020_CVPR_Workshops,
    author = {Wang, Haofan and Wang, Zifan and Du, Mengnan and Yang, Fan and Zhang, Zijian and Ding, Sirui and Mardziel, Piotr and Hu, Xia},
    title = {Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2020}
}
```

## Thanks
Utils are built on [flashtorch](https://github.com/MisaOgura/flashtorch), thanks for releasing this great work!

## Contact
If you have any questions, feel free to contact me via: `haofanw@andrew.cmu.edu`
