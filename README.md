# Nueral_networks and Deep Learning
Exploring nueral networks with deep learning models including data preprocessing and optimization. Using knowledge of machine learning and neural networks, the analysis uses the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by the charity organization.

#### Tools: Jupyter Notebooks, Python, Pandas, sklearn, tensorflow, and keras 
#### Resources: [charity_data.csv](https://github.com/emaynard10/Nueral_network_charity_analysis/files/9389981/charity_data.csv)

## Overview of the analysis: Explain the purpose of this analysis.
The purpose of this analysis is to help a charity organization determine which projects they have funded will be succesful so that they can choose the best porjecs to fund going forward. The data and prediction are complex so it uses a nueral network for flexibility. The target model performance is 75% accuracy. The code explores step for optimizing the model preformance including noisy vairiable, adding nuerons and hidden layers, changing activation functions of hidden layers, and changing the number of epochs. The third attempt at optimizing for target model performance includes using auto-optimizer with keras tuner. 

## Results
### Data Preprocessing

* What variable(s) are considered the target(s) for your model?
The variable that is considered the target in this model is the is successsful variable. 

* What variable(s) are considered to be the features for your model?
The features of the model include various features, as this was one thing that was changed for optimization. The original model run included application type, affiliation, classification, use case, organization, status, income amount, special considerations, asking amount as features. 

* What variable(s) are neither targets nor features, and should be removed from the input data?
The two vairables that were removed from the input data for the inital model run were name and EIN.

### Compiling, Training, and Evaluating the Model

* How many neurons, layers, and activation functions did you select for your neural network model, and why?
To start and to avoid overfitting which nueral networks can be prone to, the model uses relatively few number of nuerons and hidden layers. The first run there are two hidden layers and 15 and 10 nuerons. Because the rule of thumb for basic nueral network doesn't exactly apply here with a deep nueral network, which is the 2-3 times the number of inputs, the model still follows loosely. With nine features used as inputs 2 time would be 18, which was rounded down to start small and try to avoid overfitting. The subsequent runs will be used to increase the nuerons and layers. And since two hidden layers is the fewest for a deep learning model, that was the starting point. Since it doesnt't taken many hidden layers to imporve the model, again it was decided to start small. The activation functions used were relu and and output of sigmoid since it is a binary classifed model trying to answer if the charity project is successful or not. 

* Were you able to achieve the target model performance?
The analysis and various model runs were not able to acheive the target model performance of 75% accuracy.

* What steps did you take to try and increase model performance?
To increase model performance, the analysis adds both hidden layers and nuerons, increases and decreases epochs (up to 200 and down to 50), and uses other activation functions such as tanh, leaky_relu, and softmax. When these thing still did not improve model performance, the analysis applied the auto optimizer with keras_tuner. The auto-optimizer fit the best function as relu with 4 layers and 23 nuerons in the first layer over 200 epochs. tanh and sigmoid also had similar results, but no model was able to get over the best of 73.52% accuracy. Changing the features could also influence the models so the analysis tried to run the model with fewer and different features. The first attempt at optimization droppe dthe classification and application type columns and got worse moel preformance. The second attempt dropped the status and special circumstances features. The third attempt ran the auto-optimizer with the same features as the second attempt.

![Screen Shot 2022-08-22 at 10 08 13 AM](https://user-images.githubusercontent.com/99676466/185969627-335a2ac2-f14c-4293-8fff-7a4d17815959.png)

## Summary
The model was not able to get over 73% accuracy with many changes tp the nueral network. This doens't mean that it can't, this model just didnt find the right combination. That is why it would be a good idea to try some other machine learning models to compare the preformance and they might not be such a computational load so could be easier to use and interpret in the end.  After running many different combinations of deep learning models all getting to about 73% accuracy, it is recommended that a differnt model be applied to try and achieve the target model perforance of 75%. To do this the recommendation is to compare the deep learning model with both a SVM and a random forest  classifier model. The support vector models are less prone ot overfitting but can still handle complex data by classifying two groups. The advantage of the random forest classifier is that it can learn from the weak learners and be easier to interpret. It would also be worthwhile to run the model and determine which are the best features since the deep learning models ran different features without much change.
