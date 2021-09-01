# Wine quality detection

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

### Classes balance:

<img src="Stat\hist_number_of_data.png" style="zoom: 67%;" />

## - Preprocessing

Z - normalization (centering and scaling to unit variance)

gaussianization to map the features to values whose empirical comulative distrubution function is well approximated by a gaussian p.p.f

## - Features analysis

Plotting of the raw data features:





| <img src="Stat\Hist\Normalized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Normalized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Normalized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Normalized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Normalized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Normalized\hist_9.png" style="zoom: 67%;" /> |                                                              |





After gaussianization:





| <img src="Stat\Hist\Gaussianized\hist_0.png" style="zoom: 67%;" /> | <img src="Stat\Hist\Gaussianized\hist_1.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_2.png" style="zoom:67%;" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Stat\Hist\Gaussianized\hist_3.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_4.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_5.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_6.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_7.png" style="zoom:67%;" /> | <img src="Stat\Hist\Gaussianized\hist_8.png" style="zoom:67%;" /> |
| <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom:60%;" /> | <img src="Stat\Hist\Gaussianized\hist_9.png" style="zoom: 67%;" /> |                                                              |

Correlation between the features:

| All dataset<img src="Stat\HeatMaps\Gaussianized\whole_dataset.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\high_quality.png" style="zoom:67%;" /> | <img src="Stat\HeatMaps\Gaussianized\low_quality.png" style="zoom:67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

feature 5 and 6 strongly correlated -> It suggests that we may benefit from using PCA to map data to 9, 8 , 7

(very similar covariance matrixes)



##### Methodologies:

we analyze both options (#TODO: Riguardare )

- Single Fold:
- K Fold:

## - Classifier for the wine

## * MVG classifiers  
|                                       | **Single Fold** |             |             | **5-Fold**  |             |             |
| :-----------------------------------: | --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|                                       | prior=0.5       | prior=0.1   | prior=0.9   | prior=0.5   | prior=0.1   | prior=0.9   |
|   ------ **Raw Features – no PCA**    | --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | *0.346*         | 0.766       | 0.824       | *0.343*     | 0.795       | 0.906       |
|             **Diag -Cov**             | 0.527           | 0.911       | 0.937       | 0.548       | 0.962       | 0.973       |
|           **Tied Full-Cov**           | *0.355*         | 0.818       | 0.836       | *0.359*     | 0.843       | 0.827       |
|          **Tied Diag- Cov**           | 0.521           | 0.913       | 0.983       | 0.543       | 0.970       | 0.952       |
|  **Gaussianized Features – no PCA**   | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | *0.277*         | 0.791       | 0.824       | *0.306*     | 0.772       | 0.871       |
|             **Diag -Cov**             | 0.497           | 0.901       | 0.963       | 0.516       | 0.927       | 0.975       |
|           **Tied Full-Cov**           | 0.355           | 0.889       | 0.906       | 0.379       | 0.810       | 0.928       |
|          **Tied Diag- Cov**           | 0.529           | 0.913       | 0.992       | 0.543       | 0.958       | 0.995       |
|     **Raw Features – PCA (m=9)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.417           | 0.858       | 0.884       | 0.425       | 0.915       | 0.959       |
|             **Diag -Cov**             | 0.469           | 0.910       | 0.963       | 0.517       | 0.937       | 0.986       |
|           **Tied Full-Cov**           | 0.469           | 0.934       | 0.915       | 0.489       | 0.951       | 0.816       |
|          **Tied Diag- Cov**           | 0.467           | 0.934       | 0.915       | 0.491       | 0.960       | 0.921       |
| **Gaussianized Features – PCA (m=9)** | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.398           | 0.838       | 0.872       | 0.419       | 0.911       | 0.969       |
|             **Diag -Cov**             | 0.455           | 0.904       | 0.956       | 0.489       | 0.943       | 0.974       |
|           **Tied Full-Cov**           | 0.469           | 0.945       | 0.935       | 0.490       | 0.938       | 0.931       |
|          **Tied Diag- Cov**           | 0.465           | 0.909       | 0.920       | 0.495       | 0.934       | 0.938       |
|     **Raw Features – PCA (m=8)**      | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.421           | 0.949       | 0.915       | 0.477       | 0.990       | 0.948       |
|             **Diag -Cov**             | 0.552           | 0.979       | 0.968       | 0.598       | 0.985       | 0.986       |
|           **Tied Full-Cov**           | 0.504           | 0.964       | 0.887       | 0.535       | 0.982       | 0.807       |
|          **Tied Diag- Cov**           | 0.517           | 0.964       | 0.843       | 0.537       | 0.983       | 0.893       |
| **Gaussianized Features – PCA (m=8)** | -----------     | ----------- | ----------- | ----------- | ----------- | ----------- |
|            **Full - Cov**             | 0.426           | 0.902       | 0.891       | 0.471       | 0.988       | 0.946       |
|             **Diag -Cov**             | 0.526           | 0.959       | 0.920       | 0.563       | 0.982       | 0.961       |
|           **Tied Full-Cov**           | 0.497           | 0.984       | 0.911       | 0.510       | 0.973       | 0.930       |
|          **Tied Diag- Cov**           | 0.528           | 0.974       | 0.860       | 0.535       | 0.990       | 0.919       |

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

## *GMM

| <img src="Graph\GMM\GMM_Full_covariance.png" style="zoom:60%;" /> | <img src="Graph\GMM\GMM_Tied_covariance.png" style="zoom:60%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="Graph\GMM\GMM_Diagonal_covariance.png" style="zoom:60%;" /> | <img src="Graph\GMM\GMM_Tied_Diagonal_covariance.png" style="zoom:60%;" /> |



*****************************************************************************************************************

*******************************

## 

##### *** COMPARISON BETWEEN ACTUAL DCF and MIN DCF

|                                            | prior=0.5 |        | prior=0.1 |        | prior=0.9 |        |
| ------------------------------------------ | --------- | ------ | --------- | ------ | --------- | ------ |
|                                            | minDCF    | actDCF | minDCF    | actDCF | minDCF    | actDCF |
| QuadLog reg, lambda=10**-7, pi_T =0.1  Raw | 0.269     | 0.515  | 0.752     | 0.898  | 0.729     | 1.43   |
| MVG full, gaussianized, noPCA              | 0.306     | 0.327  | 0.772     | 0.834  | 0.871     | 0.991  |

Bayes Error Plot, showing the DCFs for different applications:

<img src="Graph\Error_Bayes_Plots\EBP1.png" style="zoom:80%;" />

After the threshold estimated protocol

|                                            | min DCF | act DCF (t theoretical threshold) | actDCF t estimated |
| ------------------------------------------ | ------- | --------------------------------- | ------------------ |
| **prior=0.5**                              |         |                                   |                    |
| QuadLog reg, lambda=10**-7, pi_T =0.1  Raw | 0.281   | 0.542                             | 0.297              |
| MVG full, gaussianized, noPCA              | 0.317   | 0.341                             | 0.333              |
| **prior=0.1**                              |         |                                   |                    |
| QuadLog reg, lambda=10**-7, pi_T =0.1  Raw | 0.740   | 0.890                             | 0.762              |
| MVG full, gaussianized, noPCA              | 0.804   | 0.925                             | 0.933              |
| **prior=0.9**                              |         |                                   |                    |
| QuadLog reg, lambda=10**-7, pi_T =0.1  Raw | 0.732   | 1.573                             | 0.736              |
| MVG full, gaussianized, noPCA              | 0.831   | 0.993                             | 0.898              |



## - Experimental validation


## - Conclusions

 