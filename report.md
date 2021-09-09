# Wine quality detection

## - Introduction

The dataset used for the purpose of this analysis is related to red and white variants of the Portuguese "Vinho Verde" wine and is taken from the UCI repository.

The dataset was collected to predict human wine taste preferences. As a matter of fact, it associates the physiochemical characteristics of the analyzed wine to sensory evaluations made by experts.

The original dataset consists of 10 classes (quality 1 to 10), but for this project the dataset has been binarized, collecting all wines with low quality (lower than 6) into class 0 and good quality (grater than 6) into class 0. Wines with quality 6 have been discarded to simplify the task.

The dataset contains both red and white wines that were originally separated, whereas in this analysis they have been merged.

The following analysis is aimed at exploring the characteristics of the chosen dataset to find the model that best classifies the input data. In order to fulfill this purpose, several classifiers combined with different preprocessing techniques have been exploited. The main results will be shown in this report.

### Classes balance:

The dataset has been divided into a training set and a test set.

The training set contains 613 samples belonging to the "high quality" class and 1226 belonging to the "low quality" class. On the other hand, the evaluation set contains 664 samples of class "high quality" and 1158 samples of class "low quality". Therefore, the classes are partially balanced.

The following graphs show the comparison between the number of samples of the two classes in both training and test data set. It emerges that both the number of samples and the ratio between the two classes in the different datasets is similar. 

| <img src="Stat\hist_number_of_data_Training.png" style="zoom: 67%;" /> | <img src="Stat\hist_number_of_data_Test.png" style="zoom: 67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |





### Attribute Information:

The input variables have 11 continuous features based on physiochemical tests:

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

The output variable is a discrete value representing the quality of the wine sample (0 low quality/1 high quality)

## - Preprocessing and Features analysis 

The features of the dataset refer to different types of variables and have therefore different measuring scales.

Thus, in order to compare similarities between features, the dataset has been z-normalized, through centering and scaling to unit variance the features. 

The histograms below show the distributions of the training dataset features. Orange histograms  refer to high quality wines, whereas blue histograms to low quality wines.

| <img src="Stat\Hist\Normalized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Normalized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Normalized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Normalized\hist_10.png" style="zoom: 67%;" /> |                                                              |

The analysis of the training data reveals that most of the features have an irregular distribution.

Consequently, the classification (especially of Gaussian based methods) may produce sub-optimal results.

We therefore further preprocessed our data by "Gaussianizing" the features.

The gaussianization process allows mapping the features values to ones whose empirical comulative distribution function is well approximated by a Gaussian c.d.f. To do so, the features have been mapped to a uniform distribution and then transformed  through the inverse of Gaussian cumulative distribution function.



The histograms below show the distributions of the gaussianized features.

| <img src="Stat\Hist\Gaussianized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Gaussianized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Gaussianized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Gaussianized\hist_10.png" style="zoom: 67%;" /> |                                                              |

A correlation analysis of the Gaussianized features shows that feature 5 and 6 are strongly correlated.

The heatmaps showing the Pearson correlation coefficient can be found below (****FORMULA********)

| All dataset<img src="Stat\HeatMaps\Gaussianized\whole_dataset.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\high_quality.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\low_quality.png" style="zoom:67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

This suggests that the classification may benefit from using PCA to map data to 10 or 9 uncorrelated features to reduce the number of parameters to estimate. 





## - Classification of Wine Quality features

##### Methodologies used for validation:

In order to understand which model is most promising and to assess the effects of using PCA, both single fold-validation and K-Fold cross-validation have been adopted.

At first, a single fold approach has been used, since the training with this approach is faster. Indeed, single fold approach consists in splitting the training dataset into two subsets, one (66% of the original set) for development and the other one for validation. 

Subsequently, the K-Fold approach has been used to get more robust results. In this case, the training set are iteratively split into 5 folds, 4 used for training and 1 for validation, after being shuffled. The validation scores have been eventually put together and used to compute the performance metrics. In this way, there is more data available for training and validation. 

This document focuses on the analysis of the balanced uniform prior application:

​				(prior, Cfp, Cfn) = (0.5,1,1)

However, two other unbalanced applications have been considered:

​				(prior, Cfp, Cfn) = (0.9,1,1), (prior, Cfp, Cfn) = (0.1,1,1)

??Since the classes are partially unbalanced toward the low quality class, we expect to have better results in the application biased toward the low quality class (therefore with prior 0.1) rather than the other one.



In the fist part, the main aim of the analysis was to choose the most promising approach. Therefore, the performances have been measured in terms of normalized minimum detection costs, namely the costs that would be paid by making optimal decisions for the validation set through the use of recognizers scores.

## * MVG classifiers  
|                                        | **Single Fold** |             |             | **5-Fold**  |             |             |
| :------------------------------------: | --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|                                        | prior=0.5       | prior=0.9   | prior=0.1   | prior=0.5   | prior=0.9   | prior=0.1   |
|       **Raw Features – no PCA**        | -------------   | ----------- | ----------- | ----------- | ----------- | ----------- |
|            ***Full - Cov***            | ***0.304***     | *0.812*     | *0.777*     | ***0.312*** | *0.842*     | *0.778*     |
|            ***Diag -Cov***             | *0.437*         | *0.875*     | *0.818*     | *0.420*     | *0.921*     | *0.845*     |
|          ***Tied Full-Cov***           | *0.334*         | *0.733*     | *0.779*     | *0.333*     | *0.748*     | *0.812*     |
|          ***Tied Diag- Cov***          | *0.412*         | *0.901*     | *0.832*     | *0.402*     | *0.932*     | *0.866*     |
|   **Gaussianized Features – no PCA**   | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            ***Full - Cov***            | ***0.270***     | *0.740*     | *0.807*     | ***0.306*** | *0.790*     | *0.784*     |
|            ***Diag -Cov***             | *0.456*         | *0.848*     | *0.860*     | *0.448*     | *0.914*     | *0.834*     |
|          ***Tied Full-Cov***           | *0.348*         | *0.812*     | *0.867*     | *0.354*     | *0.884*     | *0.803*     |
|          ***Tied Diag- Cov***          | *0.451*         | *0.891*     | *0.839*     | *0.451*     | *0.942*     | *0.879*     |
|     **Raw Features – PCA (m=10)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.325           | 0.814       | 0.772       | 0.321       | 0.854       | 0.797       |
|             **Diag -Cov**              | 0.375           | 0.899       | 0.821       | 0.397       | 0.924       | 0.812       |
|           **Tied Full-Cov**            | 0.341           | 0.697       | 0.806       | 0.333       | 0.760       | 0.823       |
|           **Tied Diag- Cov**           | 0.337           | 0.651       | 0.794       | 0.337       | 0.765       | 0.825       |
| **Gaussianized Features – PCA (m=10)** | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            ***Full - Cov***            | *0.312*         | *0.817*     | *0.779*     | *0.328*     | *0.862*     | *0.807*     |
|            ***Diag -Cov***             | *0.362*         | *0.747*     | *0.780*     | *0.378*     | *0.824*     | *0.804*     |
|          ***Tied Full-Cov***           | *0.324*         | *0.682*     | *0.791*     | *0.328*     | *0.753*     | *0.809*     |
|          ***Tied Diag- Cov***          | *0.335*         | *0.656*     | *0.822*     | *0.334*     | *0.782*     | *0.822*     |
|      **Raw Features – PCA (m=9)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|             **Full - Cov**             | 0.322           | 0.805       | 0.787       | 0.327       | 0.811       | 0.814       |
|             **Diag -Cov**              | 0.364           | 0.867       | 0.803       | 0.389       | 0.866       | 0.805       |
|           **Tied Full-Cov**            | 0.341           | 0.706       | 0.806       | 0.331       | 0.760       | 0.823       |
|           **Tied Diag- Cov**           | 0.343           | 0.658       | 0.799       | 0.338       | 0.765       | 0.830       |
| **Gaussianized Features – PCA (m=9)**  | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            ***Full - Cov***            | *0.300*         | *0.817*     | *0.783*     | *0.319*     | *0.814*     | *0.800*     |
|            ***Diag -Cov***             | *0.349*         | *0.764*     | *0.810*     | *0.368*     | *0.814*     | *0.804*     |
|          ***Tied Full-Cov***           | *0.331*         | *0.677*     | *0.786*     | *0.327*     | *0.752*     | *0.816*     |
|          ***Tied Diag- Cov***          | *0.337*         | *0.673*     | *0.822*     | *0.335*     | *0.783*     | *0.825*     |
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



The diagonal covariance models (both full diagonal and tied diagonal) do not give good results if compared to the other models with or without Gaussianization and PCA. These models work under the naive-bayes-assumption, according to which the different components for each class are uncorrelated. Therefore, in this case, this assumption does not produce accurate results. Applying PCA slightly improves the performance, probabily because by removing the low variances directions the whithin-class correlation decreases .

The tied full covariance model performs pretty good on the validation data, confirming the similiarity between classes shown by the correlation analysis. It performs better than the diagonal models accounting the within-class correlations. In this case, PCA does not help in producing better results.

The best-performing model is the full covariance model, which is able to account for correlations. What is more, having enough data compared to the dimensionality of the samples, the results we succeed obtaining robust results.

Gaussianization fails in improving significantly the results, although it still performs slightly better if compared to raw features.

Even if it does not help increasing much the performances, PCA can still be used to reduce the number of parameters and therefore to reduce the complexity of the model. Nevertheless, given the limited effectiveness, it will not be considered for the future analysis.

The results between single-fold and K-fold are consistent, suggesting that the amount of data is enough for validation and model training also when it comes to the single-fold set up.

None of the models produce accurate results for the unbalanced applications.

In summary, the best candidate is currently the MVG model with full covariance matrices. The chosen one is the one with Gaussianized features, since it shows slightly better results than the one with raw features, and the K-fold approach, even if it is a bit worse than the single fold result. Nonetheless, this approach provides more robust results.

Since the best-performing models are the full covariance and the tied full models (which respectively have a quadratic surface rule and a linear surface rule), the decision was to proceed by analyzing both the quadratic and the linear models.







## * LOGISTIC REGRESSION

#### ** Linear Logistic Regression

| <img src="Graph\LR\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\linear\5FoldGauss.png" style="zoom:60%;" /> |

|                                        | prior=0.5   | prior=0.1   | prior=0.9   |
| -------------------------------------- | ----------- | ----------- | ----------- |
| **Raw Features**                       | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | 0.356       | 0.835       | 0.686       |
| Log reg, lambda=10-7, pi_T =0.1        | 0.335       | 0.819       | 0.724       |
| Log reg, lambda=10-7, pi_T =0.9        | 0.370       | 0.858       | 0.646       |
| Log reg, lambda=10-7, pi_T =pi_emp_T   | 0.343       | 0.840       | 0.676       |
| **Gaussianized features**              | ----------- | ----------- | ----------- |
| Log reg, lambda=10**-7, pi_T =0.5      | 0.363       | 0.857       | 0.761       |
| Log reg, lambda=10**-7, pi_T =0.1      | 0.339       | 0.780       | 0.946       |
| Log reg, lambda=10**-7, pi_T =0.9      | 0.376       | 0.905       | 0.716       |
| Log reg, lambda=10**-7, pi_T =pi_emp_T | 0.358       | 0.831       | 0.847       |

#### ** Quadratic Logistic Regression

| <img src="Graph\LR\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\LR\quadratic\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\LR\quadratic\5FoldGauss.png" style="zoom:60%;" /> |

|                                            | prior=0.5   | prior=0.1   | prior=0.9   |
| ------------------------------------------ | ----------- | ----------- | ----------- |
| **Raw Features**                           | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.274       | 0.767       | 0.720       |
| Quad Log reg, lambda=10**-7, pi_T =0.1     | 0.276       | 0.714       | 0.747       |
| QuadLog reg, lambda=10**-7, pi_T =0.9      | 0.298       | 0.817       | 0.688       |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.275       | 0.755       | 0.725       |
| **Gaussianized features**                  | ----------- | ----------- | ----------- |
| Quad Log reg, lambda=10**-7, pi_T =0.5     | 0.296       | 0.698       | 0.666       |
| Quad Log reg, lambda=10**-7, pi_T =0.1     | 0.300       | 0.720       | 0.643       |
| Quad Log reg, lambda=10**-7, pi_T =0.9     | 0.308       | 0.761       | 0.685       |
| QuadLog reg, lambda=10**-7, pi_T =pi_emp_T | 0.295       | 0.691       | 0.662       |

## *SVM



#### ** Linear SVM

| <img src="Graph\SVM\linear\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\linear\singleFoldGauss.png" style="zoom:60%;" /> | <img src="Graph\SVM\linear\5FoldGauss.png" style="zoom:60%;" /> |

| C=0.1 COLONNE 0.9 e 0.1 (giuste ma ordine invertire l'ordine) | prior=0.5   | prior=0.9   | prior=0.1   |
| ------------------------------------------------------------ | ----------- | ----------- | ----------- |
| **Raw Features**                                             | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5                                        | 0.339       | 0.849       | 0.668       |
| SVM, C=0.1, pi_T =0.1                                        | 0.902       | 0.995       | 0.995       |
| SVM, C=0.1, pi_T =0.9                                        | 0.386       | 0.876       | 0.693       |
| SVM, C=0.1                                                   | 0.338       | 0.817       | 0.761       |
| **Gaussianized features**                                    | ----------- | ----------- | ----------- |
| SVM, C=0.1, pi_T =0.5                                        | 0.351       | 0.833       | 0.856       |
| SVM, C=0.1, pi_T =0.1                                        | 0.579       | 0.952       | 0.995       |
| SVM, C=0.1, pi_T =0.9                                        | 0.397       | 0.953       | 0.670       |
| SVM, C=0.1                                                   | 0.345       | 0.783       | 0.951       |

(Done with k fold)

#### ** Quadratic SVM

About the parameters k and c:

| <img src="Graph\SVM\Quadratic\kc\singleFoldRAW_kc.png" style="zoom:60%;" /> | <img src="Graph\SVM\Quadratic\kc\5FoldRAW_kc.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\Quadratic\kc\singleFoldGAU_kc.png" style="zoom:60%;" /> | <img src="Graph\SVM\Quadratic\kc\singleFoldGAU_kc.png" style="zoom:60%;" /> |

Best:  arancione K=0, c=1, C=0.1

| <img src="Graph\SVM\quadratic\singleFoldRAW.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldRAW.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\SVM\quadratic\singleFoldGAU.png" style="zoom:60%;" /> | <img src="Graph\SVM\quadratic\5FoldGAU.png" style="zoom:60%;" /> |

COLONNE 0.9 e 0.1 scambiate

|                                | prior=0.5   | prior=0.9   | prior=0.1   |
| ------------------------------ | ----------- | ----------- | ----------- |
| **Raw Features**               | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5      | 0.246       | 0.735       | 0.732       |
| Quad SVM, C=10, pi_T =0.1      | 0.902       | 0.995       | 0.995       |
| Quad SVM, C=10, pi_T =0.9      | 0.386       | 0.693       | 0.876       |
| Quad SVM, C=10, pi_T =pi_emp_T | 0.338       | 0.756       | 0.814       |
| **Gaussianized features**      | ----------- | ----------- | ----------- |
| Quad SVM, C=10, pi_T =0.5      | 0.246       | 0.632       | 0.709       |
| Quad SVM, C=10, pi_T =0.1      | 0.579       | 0.995       | 0.952       |
| Quad SVM, C=10, pi_T =0.9      | 0.397       | 0.670       | 0.953       |
| Quad SVM, C=10, pi_T =pi_emp_T | 0.345       | 0.942       | 0.772       |



#### ** RBF SVM #todo



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

## *GMM #todo

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

 