# Wine quality detection

## - Introduction

The dataset used for this analysis is related to red and white variants of the Portuguese "Vinho Verde" wine and is taken from the UCI repository.

The dataset was collected to predict human wine taste preferences. In fact, it associates physiochemical characteristics of the analyzed wine to sensory evaluations made by experts.

The original dataset consists of 10 classes (quality 1 to 10) , but for this project the dataset has been binarized, collecting all wines with low quality (lower than 6) into class 0 and good quality (grater than 6) into class 0. Wines with quality 6 have been discarded to simplify the task.

The dataset contains both red and white wines that originally were separated but in this analysis have been merged.

The goal of the following analysis is exploring the characteristics of the chosen dataset to find the model that best classify the input data. To do so, several classifiers combined with different preprocessing techniques have been exploited and the main results will be shown in this report.

### Classes balance:

The dataset has been split in a training set and a test set.

The training set contains 613 samples belonging to the "high quality" class and 1226 belonging to the "low quality" class. While, the evaluation set has 664 samples of class "high quality" and 1158 samples of class "low quality". Therefore the classes are partially balanced.

In the following graphs are shown the comparisons between the number of samples of the two classes in both training and test data set. It is visible that both the number of samples and the ratio between the two classes in the different data sets is close.

| <img src="Stat\hist_number_of_data_Training.png" style="zoom: 67%;" /> | <img src="Stat\hist_number_of_data_Test.png" style="zoom: 67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |





### Attribute Information:

The input variables have 11 continuous features based on physiochemical tests

0 - fixed acidity
1 - volatile acidity
2 - citric acid
3 - residual sugar
4 - chlorides
5 - free sulfur dioxide
6 - total sulfur dioxide
7 - density
8 - pH
9 - sulphates
10 - alcohol

While the output variable is a discrete value representing the quality of the wine sample (0 low quality/1 high quality)

## - Preprocessing and Features analysis 

The features of the dataset refers to different types of variables and therefore have different measuring scales.

Thus, in order to compare similarities between features, the dataset has been z- normalized, centering and scaling to unit variance the features. 

The histograms below show the distributions of the training dataset features. Orange histograms  refer to high quality wines while blue histograms to low quality wines.



| <img src="Stat\Hist\Normalized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Normalized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Normalized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Normalized\hist_10.png" style="zoom: 67%;" /> |                                                              |

The analysis of the training data reveals that most of the features have an irregular distribution.

For this reason, the classification (especially of Gaussian based methods) may produce sub-optimal results.

We therefore further preprocessed our data by "gaussianizing" the features.

The gaussianization process allows mapping the features values to ones whose empirical comulative distribution function is well approximated by a Gaussian c.d.f. To do so, the features have been mapped to a uniform distribution and then transformed  through the inverse of Gaussian cumulative distribution function.



The histograms below show the distributions of the gaussianized features.

| <img src="Stat\Hist\Gaussianized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Gaussianized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Gaussianized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Gaussianized\hist_10.png" style="zoom: 67%;" /> |                                                              |

A correlation analysis of the Gaussianized features shows that feature 5 and 6 are strongly correlated.

Below can be found the heatmaps showing the Pearson correlation coefficient (****FORMULA********)

| All dataset<img src="Stat\HeatMaps\Gaussianized\whole_dataset.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\high_quality.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\low_quality.png" style="zoom:67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

This suggests the classification may benefit from using PCA to map data to 10 or 9 uncorrelated features to reduce the number of parameters to estimate. 





## - Classification of Wine Quality features

##### Methodologies used for validation:

To understand which model is most promising and to assess the effects of using PCA, both single fold-validation and K-Fold cross-validation have been adopted.

At first, it has been used a single fold approach where the training dataset has been split in two subsets: one (66% of the original set) for development and the other one for validation. This because the training with this approach is faster.

In a second moment, the K-Fold approach has been used to get more robust result.

This because, in the K- Fold, iteratively the training set has been split into 5 folds, 4 used for training and 1 for validation, after being shuffled. At the end, the validation scores are put together and used to compute the performance metrics. In this way, there is more data available for training and validation. 

## * MVG classifiers  
|                                        | **Single Fold** |             |             | **5-Fold**  |             |             |
| :------------------------------------: | --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|                                        | prior=0.5       | prior=0.9   | prior=0.1   | prior=0.5   | prior=0.9   | prior=0.1   |
|       **Raw Features – no PCA**        | -------------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.304           | 0.812       | 0.777       | 0.312       | 0.842       | 0.778       |
|             **Diag -Cov**              | 0.437           | 0.875       | 0.818       | 0.420       | 0.921       | 0.845       |
|           **Tied Full-Cov**            | 0.334           | 0.733       | 0.779       | 0.333       | 0.748       | 0.812       |
|           **Tied Diag- Cov**           | 0.412           | 0.901       | 0.832       | 0.402       | 0.932       | 0.866       |
|   **Gaussianized Features – no PCA**   | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.270           | 0.740       | 0.807       | 0.306       | 0.790       | 0.784       |
|             **Diag -Cov**              | 0.456           | 0.848       | 0.860       | 0.448       | 0.914       | 0.834       |
|           **Tied Full-Cov**            | 0.348           | 0.812       | 0.867       | 0.354       | 0.884       | 0.803       |
|           **Tied Diag- Cov**           | 0.451           | 0.891       | 0.839       | 0.451       | 0.942       | 0.879       |
|     **Raw Features – PCA (m=10)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.325           | 0.814       | 0.772       | 0.321       | 0.854       | 0.797       |
|             **Diag -Cov**              | 0.375           | 0.899       | 0.821       | 0.397       | 0.924       | 0.812       |
|           **Tied Full-Cov**            | 0.341           | 0.697       | 0.806       | 0.333       | 0.760       | 0.823       |
|           **Tied Diag- Cov**           | 0.337           | 0.651       | 0.794       | 0.337       | 0.765       | 0.825       |
| **Gaussianized Features – PCA (m=10)** | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.312           | 0.817       | 0.779       | 0.328       | 0.862       | 0.807       |
|             **Diag -Cov**              | 0.362           | 0.747       | 0.780       | 0.378       | 0.824       | 0.804       |
|           **Tied Full-Cov**            | 0.324           | 0.682       | 0.791       | 0.328       | 0.753       | 0.809       |
|           **Tied Diag- Cov**           | 0.335           | 0.656       | 0.822       | 0.334       | 0.782       | 0.822       |
|      **Raw Features – PCA (m=9)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.322           | 0.805       | 0.787       | 0.327       | 0.811       | 0.814       |
|             **Diag -Cov**              | 0.364           | 0.867       | 0.803       | 0.389       | 0.866       | 0.805       |
|           **Tied Full-Cov**            | 0.341           | 0.706       | 0.806       | 0.331       | 0.760       | 0.823       |
|           **Tied Diag- Cov**           | 0.343           | 0.658       | 0.799       | 0.338       | 0.765       | 0.830       |
| **Gaussianized Features – PCA (m=9)**  | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.300           | 0.817       | 0.783       | 0.319       | 0.814       | 0.800       |
|             **Diag -Cov**              | 0.349           | 0.764       | 0.810       | 0.368       | 0.814       | 0.804       |
|           **Tied Full-Cov**            | 0.331           | 0.677       | 0.786       | 0.327       | 0.752       | 0.816       |
|           **Tied Diag- Cov**           | 0.337           | 0.673       | 0.822       | 0.335       | 0.783       | 0.825       |
|      **Raw Features – PCA (m=8)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.316           | 0.858       | 0.779       | 0.349       | 0.858       | 0.824       |
|             **Diag -Cov**              | 0.387           | 0.913       | 0.867       | 0.394       | 0.907       | 0.828       |
|           **Tied Full-Cov**            | 0.379           | 0.685       | 0.860       | 0.377       | 0.867       | 0.853       |
|           **Tied Diag- Cov**           | 0.375           | 0.685       | 0.861       | 0.376       | 0.873       | 0.840       |
| **Gaussianized Features – PCA (m=8)**  | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.311           | 0.858       | 0.767       | 0.340       | 0.820       | 0.808       |
|             **Diag -Cov**              | 0.372           | 0.730       | 0.800       | 0.400       | 0.888       | 0.801       |
|           **Tied Full-Cov**            | 0.373           | 0.685       | 0.854       | 0.372       | 0.844       | 0.845       |
|           **Tied Diag- Cov**           | 0.367           | 0.745       | 0.852       | 0.373       | 0.869       | 0.854       |


notes:

- gaussianization does not improve a lot
- best one in any cases is full - cov
- best - > full mvg, tied full cov
- PCA peggiora!
- gaussianized slightly improves the results



## * LOGISTIC REGRESSION

#### ** Linear Logistic Regression

| <img src="Graph\LR\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldGauss.png" style="zoom:60%;" /> |

|                                        | prior=0.5   | prior=0.1   | prior=0.9   |
| -------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                       | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | *0.377*     | 0.864       | 0.795       |
| Log reg, lambda=10**-7, pi_T =0.9      | 0.390       | 0.903       | 0.742       |
| Log reg, lambda=10**-7, pi_T =0.1      | *0.362*     | 0.852       | 0.828       |
| Log reg, lambda=10**-7, pi_T =pi_emp_T | 0.368       | 0.862       | 0.792       |
| **Gaussianized features**              | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | 0.371       | 0.854       | 0.818       |
| Log reg, lambda=10**-7, pi_T =0.9      | 0.375       | 0.914       | 0.827       |
| Log reg, lambda=10**-7, pi_T =0.1      | 0.372       | 0.791       | 0.986       |
| Log reg, lambda=10**-7, pi_T =pi_emp_T | 0.378       | 0.840       | 0.882       |

#### ** Quadratic Logistic Regression

| <img src="Graph\LR\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\quadratic\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldGauss.png" style="zoom:60%;" /> |

|                                            | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------------------------ | ----------- | ----------- | ----------- |
| **Raw Features**                           | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.274       | 0.743       | 0.752       |
| Quad Log reg, lambda=10**-7, pi_T =0.9     | 0.301       | 0.781       | 0.753       |
| QuadLog reg, lambda=10**-7, pi_T =0.1      | 0.269       | 0.752       | 0.729       |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.269       | 0.746       | 0.743       |
| **Gaussianized features**                  | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.300       | 0.749       | 0.692       |
| Quad Log reg, lambda=10**-7, pi_T =0.9     | 0.302       | 0.811       | 0.632       |
| Quad Log reg, lambda=10**-7, pi_T =0.1     | 0.313       | 0.714       | 0.731       |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.296       | 0.739       | 0.724       |

## *SVM



#### ** Linear SVM

| <img src="Graph\SVM\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldGauss.png" style="zoom:60%;" /> |

|                           | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**          | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5     | 0.367       | 0.874       | 0.802       |
| SVM, C=0.1, pi_T =0.1     | 0.734       | 1.0         | 1.014       |
| SVM, C=0.1, pi_T =0.9     | 0.412       | 0.951       | 0.793       |
| SVM, C=0.1                | 0.367       | 0.834       | 0.849       |
| **Gaussianized features** | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5     | 0.368       | 0.853       | 0.911       |
| SVM, C=0.1, pi_T =0.1     | 0.694       | 0.949       | 0.997       |
| SVM, C=0.1, pi_T =0.9     | 0.598       | 0.976       | 0.934       |
| SVM, C=0.1                | 0.367       | 0.806       | 0.984       |

(Done with k fold)

#### ** Quadratic SVM

About the parameters k and c:

| <img src="Graph\SVM\Quadratic\kc\singleFoldRAW_kc.png" style="zoom:60%;" /> | <img src="Graph\SVM\Quadratic\kc\5FoldRAW_kc.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\Quadratic\kc\singleFoldGAU_kc.png" style="zoom:60%;" /> | <img src="Graph\SVM\Quadratic\kc\singleFoldGAU_kc.png" style="zoom:60%;" /> |

Best: 

RAW 

K=0, c=1, C=10

K=1, c=1, C=10 (prova anche 100)

GAUSSIANIZED

K=1, c=1, C=0.1



|                                         | prior=0.5   | prior=0.1   | prior=0.9   |
| --------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                        | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0      | 0.241       | 0.762       | 0.760       |
| Quad SVM, C=10, pi_T =pi_emp_t,c=1,K=0  | 0.362       | 0.843       | 0.841       |
| Quad SVM, C=100, pi_T=0.5, c=1, K=0     | 0.248       | 0.764       | 0.734       |
| Quad SVM, C=0.1, pi_T=0.5, c=1, K=1     | 0.263       | 0.771       | 0.702       |
| **Gaussianized features**               | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0      | 0.261       | 0.714       | 0.723       |
| Quad SVM, C=10, pi_T =pi_emp_t, c=1,K=0 | 0.370       | 0.972       | 0.803       |
| Quad SVM, C=100, pi_T=0.5, c=1, K=0     | 0.270       | 0.721       | 0.740       |
| Quad SVM, C=0.1, pi_T=0.5, c=1, K=1     | 0.251       | 0.709       | 0.709       |

| <img src="Graph\SVM\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\quadratic\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldGauss.png" style="zoom:60%;" /> |

#### ** RBF SVM



| <img src="Graph\SVM\RBF\5FoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\RBF\5FoldGauss.png" style="zoom:60%;" /> |
| ---------------------------------------------------------- | ------------------------------------------------------------ |

loglam = 0, C=1, 0.5



|                                     | prior=0.5   | prior=0.1   | prior=0.9   |
| ----------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                    | ----------- | ----------- | ----------- |
| RBF SVM, C=1, lam=1, pi_T =0.5      | 0.241       | 0.619       | 0.537       |
| RBF SVM, C=1, lam=1, pi_T =0.1      | 0.380       | 0.709       | 0.541       |
| RBF SVM, C=1, lam=1, pi_T =0.9      | 0.304       | 0.592       | 0.587       |
| RBF SVM, C=1, lam=1, pi_T =pi_emp_T | 0.250       | 0.641       | 0.522       |
| **Gaussianized features**           | ----------- | ----------- | ----------- |
| RBF SVM, C=1, lam=1, pi_T =0.5      | 0.248       | 0.610       | 0.542       |
| RBF SVM, C=1, lam=1, pi_T =0.1      | 0.360       | 0.646       | 0.557       |
| RBF SVM, C=1, lam=1, pi_T =0.9      | 0.284       | 0.597       | 0.567       |
| RBF SVM, C=1, lam=1, pi_T =pi_emp_T | 0.261       | 0.605       | 0.544       |

|                                       | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                      | ----------- | ----------- | ----------- |
| RBF SVM, C=0.5, lam=1, pi_T =0.5      | 0.243       | 0.690       | 0.590       |
| RBF SVM, C=0.5, lam=1, pi_T =0.1      | 0.443       | 0.785       | 0.623       |
| RBF SVM, C=0.5, lam=1, pi_T =0.9      | 0.343       | 0.599       | 0.630       |
| RBF SVM, C=0.5, lam=1, pi_T =pi_emp_T | 0.261       | 0.725       | 0.584       |
| **Gaussianized features**             | ----------- | ----------- | ----------- |
| RBF SVM, C=0.5, lam=1, pi_T =0.5      | 0.234       | 0.623       | 0.566       |
| RBF SVM, C=0.5, lam=1, pi_T =0.1      | 0.397       | 0.699       | 0.627       |
| RBF SVM, C=0.5, lam=1, pi_T =0.9      | 0.334       | 0.546       | 0.588       |
| RBF SVM, C=0.5, lam=1, pi_T =pi_emp_T | 0.265       | 0.659       | 0.575       |

## *GMM

| <img src="Graph\GMM\GMM_Full_covariance.png" style="zoom:60%;" /> | <img src="Graph\GMM\GMM_Tied_covariance.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\GMM\GMM_Diagonal_covariance.png" style="zoom:60%;" /> | <img src="Graph\GMM\GMM_Tied_Diagonal_covariance.png" style="zoom:60%;" /> |

| Num components:            | 1     | 2     | 4     | 8     | 16    | 32    |
| -------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| ----------**RAW**          |       |       |       |       |       |       |
| Full                       | 0.343 | 0.321 | 0.321 | 0.296 | 0.309 | 0.314 |
| Diag                       | 0.548 | 0.469 | 0.393 | 0.374 | 0.334 | 0.325 |
| Tied                       | 0.343 | 0.343 | 0.315 | 0.306 | 0.299 | 0.302 |
| Tied Diag                  | 0.548 | 0.507 | 0.424 | 0.391 | 0.374 | 0.321 |
| **----------Gaussianized** |       |       |       |       |       |       |
| Full                       | 0.306 | 0.317 | 0.299 | 0.305 | 0.326 | 0.337 |
| Diag                       | 0.516 | 0.393 | 0.362 | 0.355 | 0.328 | 0.325 |
| Tied                       | 0.306 | 0.306 | 0.309 | 0.311 | 0.305 | 0.287 |
| Tied Diag                  | 0.516 | 0.437 | 0.370 | 0.377 | 0.334 | 0.318 |



*****************************************************************************************************************

*******************************

## 

RBF SVM, C=0.5, lam=1, pi_T =0.5 gaussianized features

Quad SVM, C=10, pi_T =0.5, c=1,K=0 raw features 



##### *** COMPARISON BETWEEN ACTUAL DCF and MIN DCF

|                                                        | prior=0.5 |        | prior=0.1 |        | prior=0.9 |        |
| ------------------------------------------------------ | --------- | ------ | --------- | ------ | --------- | ------ |
|                                                        | minDCF    | actDCF | minDCF    | actDCF | minDCF    | actDCF |
| RBF SVM, C=0.5, lam=1, pi_T =0.5 gaussianized features | 0.234     | 0.239  | 0.566     | 1.0    | 0.623     | 1.0    |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0 raw features        | 0.278     | 0.288  | 0.782     | 0.819  | 0.753     | 0.792  |

Bayes Error Plot, showing the DCFs for different applications:

<img src="Graph\Error_Bayes_Plots\EBP1.png" style="zoom:80%;" />

After the threshold estimated protocol

|                                                        | min DCF | act DCF (t theoretical threshold) | actDCF t estimated |
| ------------------------------------------------------ | ------- | --------------------------------- | ------------------ |
| **prior=0.5**                                          |         |                                   |                    |
| RBF SVM, C=0.5, lam=1, pi_T =0.5 gaussianized features | 0.244   | 0.247                             | 0.244              |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0 raw features        | 0.279   | 0.294                             | 0.295              |
| **prior=0.1**                                          |         |                                   |                    |
| RBF SVM, C=0.5, lam=1, pi_T =0.5 gaussianized features | 0.581   | 1.0                               | 0.613              |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0 raw features        | 0.800   | 0.854                             | 0.844              |
| **prior=0.9**                                          |         |                                   |                    |
| RBF SVM, C=0.5, lam=1, pi_T =0.5 gaussianized features | 0.614   | 1.0                               | 0.615              |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0 raw features        | 0.747   | 0.772                             | 0.777              |



## - Experimental results

#### MVG :

|                                       | 66% Data      |             |             | 100% Data   |             |             |
| :-----------------------------------: | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|                                       | prior=0.5     | prior=0.1   | prior=0.9   | prior=0.5   | prior=0.1   | prior=0.9   |
|       **Raw Features – no PCA**       | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.345         |             |             | 0.355       |             |             |
|             **Diag -Cov**             | 0.468         |             |             | 0.466       |             |             |
|           **Tied Full-Cov**           | 0.334         |             |             | 0.333       |             |             |
|          **Tied Diag- Cov**           | 0.468         |             |             | 0.461       |             |             |
|  **Gaussianized Features – no PCA**   | -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.332         |             |             | 0.334       |             |             |
|             **Diag -Cov**             | 0.432         |             |             | 0.431       |             |             |
|           **Tied Full-Cov**           | 0.345         |             |             | 0.343       |             |             |
|          **Tied Diag- Cov**           | 0.453         |             |             | 0.446       |             |             |
|     **Raw Features – PCA (m=9)**      | -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.404         |             |             | 0.408       |             |             |
|             **Diag -Cov**             | 0.476         |             |             | 0.470       |             |             |
|           **Tied Full-Cov**           | 0.421         |             |             | 0.413       |             |             |
|          **Tied Diag- Cov**           | 0.418         |             |             | 0.415       |             |             |
| **Gaussianized Features – PCA (m=9)** | -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.375         |             |             | 0.377       |             |             |
|             **Diag -Cov**             | 0.431         |             |             | 0.427       |             |             |
|           **Tied Full-Cov**           | 0.420         |             |             | 0.416       |             |             |
|          **Tied Diag- Cov**           | 0.423         |             |             | 0.421       |             |             |
|     **Raw Features – PCA (m=8)**      | -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.447         |             |             | 0.435       |             |             |
|             **Diag -Cov**             | 0.525         |             |             | 0.527       |             |             |
|           **Tied Full-Cov**           | 0.473         |             |             | 0.476       |             |             |
|          **Tied Diag- Cov**           | 0.475         |             |             | 0.479       |             |             |
| **Gaussianized Features – PCA (m=8)** | -----------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.418         |             |             | 0.424       |             |             |
|             **Diag -Cov**             | 0.499         |             |             | 0.498       |             |             |
|           **Tied Full-Cov**           | 0.441         |             |             | 0.454       |             |             |
|          **Tied Diag- Cov**           | 0.482         |             |             | 0.489       |             |             |

#### LR

|                                        | prior=0.5   | prior=0.1   | prior=0.9   |
| -------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                       | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | 0.348       |             |             |
| Log reg, lambda=10**-7, pi_T =0.9      | 0.339       |             |             |
| Log reg, lambda=10**-7, pi_T =0.1      | 0.379       |             |             |
| Log reg, lambda=10**-7, pi_T =pi_emp_T | 0.368       |             |             |
| **Gaussianized features**              | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | 0.365       |             |             |
| Log reg, lambda=10**-7, pi_T =0.9      | 0.327       |             |             |
| Log reg, lambda=10**-7, pi_T =0.1      | 0.381       |             |             |
| Log reg, lambda=10**-7, pi_T =pi_emp_T | 0.378       |             |             |

#### Quad LR

|                                            | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------------------------ | ----------- | ----------- | ----------- |
| **Raw Features**                           | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.282       |             |             |
| Quad Log reg, lambda=10**-7, pi_T =0.1     | 0.300       |             |             |
| QuadLog reg, lambda=10**-7, pi_T =0.9      | 0.274       |             |             |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.286       |             |             |
| **Gaussianized features**                  | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.311       |             |             |
| Quad Log reg, lambda=10**-7, pi_T =0.1     | 0.316       |             |             |
| Quad Log reg, lambda=10**-7, pi_T =0.9     | 0.305       |             |             |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.305       |             |             |

#### SVM

|                           | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**          | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5     | 0.349       |             |             |
| SVM, C=0.1, pi_T =0.1     | 0.900       |             |             |
| SVM, C=0.1, pi_T =0.9     | 0.415       |             |             |
| SVM, C=0.1                | 0.332       |             |             |
| **Gaussianized features** | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5     | 0.350       |             |             |
| SVM, C=0.1, pi_T =0.1     | 0.853       |             |             |
| SVM, C=0.1, pi_T =0.9     | 0.401       |             |             |
| SVM, C=0.1                | 0.332       |             |             |

#### QUAD SVM

|                                         | prior=0.5   | prior=0.1   | prior=0.9   |
| --------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                        | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0      | 0.283       |             |             |
| Quad SVM, C=10, pi_T =pi_emp_t,c=1,K=0  | 0.335       |             |             |
| Quad SVM, C=100, pi_T=0.5, c=1, K=0     | 0.294       |             |             |
| Quad SVM, C=0.1, pi_T=0.5, c=1, K=1     | 0.287       |             |             |
| **Gaussianized features**               | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5, c=1,K=0      | 0.311       |             |             |
| Quad SVM, C=10, pi_T =pi_emp_t, c=1,K=0 | 0.336       |             |             |
| Quad SVM, C=100, pi_T=0.5, c=1, K=0     | 0.311       |             |             |
| Quad SVM, C=0.1, pi_T=0.5, c=1, K=1     | 0.332       |             |             |

#### RBF

|                                     | prior=0.5   | prior=0.1   | prior=0.9   |
| ----------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                    | ----------- | ----------- | ----------- |
| RBF SVM, C=1, lam=1, pi_T =0.5      | 0.330       |             |             |
| RBF SVM, C=1, lam=1, pi_T =0.1      | 0.493       |             |             |
| RBF SVM, C=1, lam=1, pi_T =0.9      | 0.310       |             |             |
| RBF SVM, C=1, lam=1, pi_T =pi_emp_T | 0.340       |             |             |
| **Gaussianized features**           | ----------- | ----------- | ----------- |
| RBF SVM, C=1, lam=1, pi_T =0.5      | 0.302       |             |             |
| RBF SVM, C=1, lam=1, pi_T =0.1      | 0.461       |             |             |
| RBF SVM, C=1, lam=1, pi_T =0.9      | 0.316       |             |             |
| RBF SVM, C=1, lam=1, pi_T =pi_emp_T | 0.331       |             |             |

|                                       | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                      | ----------- | ----------- | ----------- |
| RBF SVM, C=0.5, lam=1, pi_T =0.5      | 0.323       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =0.1      | 0.566       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =0.9      | 0.335       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =pi_emp_T | 0.348       |             |             |
| **Gaussianized features**             | ----------- | ----------- | ----------- |
| RBF SVM, C=0.5, lam=1, pi_T =0.5      | 0.284       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =0.1      | 0.507       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =0.9      | 0.355       |             |             |
| RBF SVM, C=0.5, lam=1, pi_T =pi_emp_T | 0.311       |             |             |

#### GMM

| Num components:            | 1     | 2     | 4     | 8     | 16    | 32    |
| -------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| ----------**RAW**          |       |       |       |       |       |       |
| Full                       | 0.355 | 0.316 | 0.339 | 0.342 | 0.369 | 0.394 |
| Diag                       | 0.466 | 0.427 | 0.362 | 0.378 | 0.359 | 0.339 |
| Tied                       | 0.355 | 0.355 | 0.319 | 0.310 | 0.300 | 0.302 |
| Tied Diag                  | 0.466 | 0.429 | 0.410 | 0.353 | 0.318 | 0.326 |
| **----------Gaussianized** |       |       |       |       |       |       |
| Full                       | 0.334 | 0.326 | 0.312 | 0.334 | 0.373 | 0.397 |
| Diag                       | 0.431 | 0.343 | 0.303 | 0.360 | 0.323 | 0.319 |
| Tied                       | 0.334 | 0.334 | 0.342 | 0.287 | 0.295 | 0.254 |
| Tied Diag                  | 0.431 | 0.389 | 0.331 | 0.361 | 0.330 | 0.323 |







## - Conclusions

 