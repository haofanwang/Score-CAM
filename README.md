### Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks

Recently, increasing attention has been drawn to the internal mechanisms of convolutional neural networks, and the reason why the network makes specific decisions. In this paper, we develop a novel post-hoc visual explanation method called Score-CAM based on class activation mapping. Unlike previous class activation mapping based approaches, Score-CAM gets rid of the dependence on gradients by obtaining the weight of each activation map through its forward passing score on target class, the final result is obtained by a linear combination of weights and activation maps. We demonstrate that Score-CAM achieves better visual performance and fairness for interpreting the decision making process. Our approach outperforms previous methods on both recognition and localization tasks, it also passes the sanity check. We also indicate its application as debugging tools.

Paper: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://haofanwang.github.io/documents/Score-CAM.pdf) to appear at IEEE CVPR 2020 Workshop on Fair, Data Efficient and Trusted Computer Vision.

## Update
**`2020.4.13`**: First version of Score-CAM code has been released. More implementations will be added later.

## Milestone
* Support for Colab notebook.
* Support for faster version of Score-CAM.
* Support for pre-trained model in Pytorch.
* Support for self-defined model in Pytorch.
* Add visualization result and quantitive evaluation.
* Support for object localization task.

## Implement Score-CAM into popular visualization tools.
It would be very appreciated for implementing Score-CAM for other popular projects, if any of you are interested.
* Implement in [pytorch/captum](https://github.com/pytorch/captum)
* Implement in [sicara/tf-explain](https://github.com/sicara/tf-explain)
* Implement in [PAIR-code/saliency](https://github.com/PAIR-code/saliency)
* Implement in [experiencor/deep-viz-keras](https://github.com/experiencor/deep-viz-keras)
* Implement in [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

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
@article{wang2019score,
  title={Score-CAM: Improved Visual Explanations Via Score-Weighted Class Activation Mapping},
  author={Wang, Haofan and Du, Mengnan and Yang, Fan and Zhang, Zijian},
  journal={arXiv preprint arXiv:1910.01279},
  year={2019}
}
```

## Thanks
Utils are built on [flashtorch](https://github.com/MisaOgura/flashtorch), thanks for releasing this great work!

## Contact
If you have any questions, feel free to contact me via: `haofanw@andrew.cmu.edu`
