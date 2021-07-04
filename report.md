# Banknode authentication

## - Introduction

### Data Set Information:

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.


### Attribute Information:

1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)

## - Features analysis

Plotting of the features:

![](Stat/Hist/hist_0.png)
![](Stat/Hist/hist_1.png)
![](Stat/Hist/hist_2.png)
![](Stat/Hist/hist_3.png)

Correlation between the features:
![](Stat/HeatMaps/whole_dataset.png)
![](Stat/HeatMaps/forged_dataset.png)
![](Stat/HeatMaps/authentic_dataset.png)

## - Classifier for the banknote task

## - Experimental validation

## - Conclusions