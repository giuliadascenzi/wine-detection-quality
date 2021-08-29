

# Banknode authentication

## - Introduction

### Data Set Information:

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

### Attribute Information:

Input variables (based on physicochemical tests):

1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol

Output variable (based on sensory data):
12 - quality (score between 0 and 10)

## - Features analysis

Plotting of the raw data features:





| <img src="Stat\Hist\Normalized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Normalized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Normalized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Normalized\hist_9.png" style="zoom: 67%;" /> |                                                              |



After gaussialization





| <img src="Stat\Hist\Gaussianized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Gaussianized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Gaussianized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom: 67%;" /> |                                                              |









Correlation between the features:

| <img src="Stat\HeatMaps\Gaussianized\whole_dataset.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\high_quality.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\low_quality.png" style="zoom:67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

feature 5 and 6 strongly correlated

very similar covariance matrixes

## - Classifier for the wine

## * MVG classifiers
|                                       | **Single Fold**                                              |                                                              |                                                              | **5-Fold**                                                   |                                                              |                                                              |
| :-----------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                       | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png) |
|   ------ **Raw Features – no PCA**    | ---------------                                              | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.346                                                        | 0.766                                                        | 0.824                                                        | 0.343                                                        | 0.795                                                        | 0.906                                                        |
|             **Diag -Cov**             | 0.527                                                        | 0.911                                                        | 0.937                                                        | 0.548                                                        | 0.962                                                        | 0.973                                                        |
|           **Tied Full-Cov**           | 0.355                                                        | 0.818                                                        | 0.836                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |
|  **Gaussianized Features – no PCA**   | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.277                                                        | 0.791                                                        | 0.824                                                        | 0.306                                                        | 0.772                                                        | 0.871                                                        |
|             **Diag -Cov**             | 0.497                                                        | 0.901                                                        | 0.963                                                        | 0.516                                                        | 0.927                                                        | 0.975                                                        |
|           **Tied Full-Cov**           | 0.355                                                        | 0.889                                                        | 0.906                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |
|     **Raw Features – PCA (m=9)**      | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.455                                                        | 0.966                                                        | 0.884                                                        | 0.495                                                        | 0.980                                                        | 0.975                                                        |
|             **Diag -Cov**             | 0.541                                                        | 0.963                                                        | 0.947                                                        | 0.570                                                        | 1.0                                                          | 0.988                                                        |
|           **Tied Full-Cov**           | 0.55                                                         | 0.974                                                        | 0.939                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |
| **Gaussianized Features – PCA (m=9)** | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.300                                                        | 0.820                                                        | 0.858                                                        | 0.336                                                        | 0.805                                                        | 0.898                                                        |
|             **Diag -Cov**             | 0.392                                                        | 0.834                                                        | 0.899                                                        | 0.407                                                        | 0.884                                                        | 0.899                                                        |
|           **Tied Full-Cov**           | 0.421                                                        | 0.897                                                        | 0.903                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |
|     **Raw Features – PCA (m=8)**      | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.490                                                        | 0.970                                                        | 0.930                                                        | 0.537                                                        | 0.998                                                        | 0.951                                                        |
|             **Diag -Cov**             | 0.606                                                        | 0.992                                                        | 0.966                                                        | 0.611                                                        | 1.0                                                          | 0.963                                                        |
|           **Tied Full-Cov**           | 0.585                                                        | 0.963                                                        | 0.947                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |
| **Gaussianized Features – PCA (m=8)** | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  | -----------                                                  |
|            **Full - Cov**             | 0.345                                                        | 0.870                                                        | 0.867                                                        | 0.385                                                        | 0.932                                                        | 0.892                                                        |
|             **Diag -Cov**             | 0.427                                                        | 0.945                                                        | 0.927                                                        | 0.458                                                        | 0.982                                                        | 0.900                                                        |
|           **Tied Full-Cov**           | 0.434                                                        | 0.987                                                        | 0.853                                                        | 0.359                                                        | 0.843                                                        | 0.827                                                        |
|          **Tied Diag- Cov**           | 0.521                                                        | 0.913                                                        | 0.983                                                        | 0.543                                                        | 0.970                                                        | 0.952                                                        |


********************************************************************

## * LOGISTIC REGRESSION

#### ** Linear Logistic Regression

| <img src="Graph\LR\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldGauss.png" style="zoom:60%;" /> |

|                                                              | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png) | ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Raw Features**                                             | -----------                                                  | -----------                                                  | -----------                                                  |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)**)** | 0.559                                                        | 0.969                                                        | 0.934                                                        |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)**)** | 0.584                                                        | 0.977                                                        | 0.917                                                        |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png)**)** | 0.556                                                        | 0.963                                                        | 0.952                                                        |
| **Gaussianized features**                                    | -----------                                                  | -----------                                                  | -----------                                                  |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)**)** | 0.371                                                        | 0.854                                                        | 0.818                                                        |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)**)** | 0.375                                                        | 0.914                                                        | 0.827                                                        |
| Log Reg (**λ****=** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) **,** ![img](file:///C:/Users/Utente/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png)**)** | 0.372                                                        | 0.791                                                        | 0.986                                                        |

#### ** Quadratic Logistic Regression

| <img src="Graph\LR\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\quadratic\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldGauss.png" style="zoom:60%;" /> |

## * SVM

#### ** Linear SVM

| <img src="Graph\SVM\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldGauss.png" style="zoom:60%;" /> |



#### ** Quadratic SVM

| <img src="Graph\SVM\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\quadratic\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldGauss.png" style="zoom:60%;" /> |

#### ** RBF SVM

|      |      |
| ---- | ---- |
|      |      |

## - Experimental validation


## - Conclusions

