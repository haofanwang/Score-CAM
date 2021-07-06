## Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
To appear at IEEE [CVPR 2020 Workshop on Fair, Data Efficient and Trusted Computer Vision](https://fadetrcv.github.io).

<img src="https://github.com/haofanwang/Score-CAM/blob/master/pics/comparison.png" width="100%" height="100%">

In this paper, we develop a novel post-hoc visual explanation method called Score-CAM based on class activation mapping. Score-CAM is a gradient-free visualization method, extended from Grad-CAM and Grad-CAM++. It achieves better visual performance (state-of-the-art) and fairness for interpreting the decision making process. 

Paper: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf)

Demo: You can run an example via [Colab](https://colab.research.google.com/drive/1m1VAhKaO7Jns5qt5igfd7lSVZudoKmID?usp=sharing)

## Update

**`2021.4.03`**: Merged into [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (2.8K Stars).

**`2020.8.18`**: Merged into [PaddlePaddle/InterpretDL](https://github.com/PaddlePaddle/InterpretDL), a toolkit for PaddlePaddle models.

**`2020.7.11`**: Merged into [keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis), written in Tensorflow.

**`2020.5.11`**: Merged into [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (5.7K Stars).

**`2020.3.24`**: Merged into [frgfm/torch-cam](https://github.com/frgfm/torch-cam), a wonderful library that supports multiple CAM-based methods.


## Citation
If you find this work or code is helpful in your research, please cite our work:
```
@inproceedings{wang2020score,
  title={Score-CAM: Score-weighted visual explanations for convolutional neural networks},
  author={Wang, Haofan and Wang, Zifan and Du, Mengnan and Yang, Fan and Zhang, Zijian and Ding, Sirui and Mardziel, Piotr and Hu, Xia},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={24--25},
  year={2020}
}
```

## Thanks
Utils are built on [flashtorch](https://github.com/MisaOgura/flashtorch), thanks for releasing this great work!

## Contact
If you have any questions, feel free to open an issue or directly contact me via: `haofanw@andrew.cmu.edu`
