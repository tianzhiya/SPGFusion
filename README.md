# SPGFusion
Code of SPGFusion: a unsupervised image fusion network for multiple image fusion tasks, including multi-modal (VIS-IR) image fusion.

## Tips:<br>
Due to file size issues, the training set has been removed from the code and the MSRS dataset can be downloaded here: https://github.com/Linfeng-Tang/MSRS
Place the downloaded training dataset under: MSRS/Visible/train path.

## To Train
Run "**--isTrain=1 python main.py**" to train your model.
The training data are selected from the MSRS dataset. 

## To Test
Run "**--isTrain=0 python main.py**" to test the model.
The images generated by the test will be placed under the MSRS/val/MSRS path.

If this work is helpful to you, please cite it as:
```
@article{SPGFusion,
  title={SPGFUSION: A SEMANTIC PRIOR GUIDED INFRARED AND VISIBLE IMAGE FUSION
NETWORK},
author={Quanquan Xiao ,Haiyan jin,Haonan Su,etc},
}
```
If you have any question, please email to me (1211211001@stu.xaut.edu.cn).
