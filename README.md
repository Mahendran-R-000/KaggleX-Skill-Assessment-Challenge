# Car Price Prediction: KaggleX Skill Assessment Challenge

## KaggleX Skill Assessment Challenge
This project is part of the KaggleX Skill Assessment Challenge. The challenge aims to validate the hands-on experience of participants in data science. The leaderboard for this challenge does not reflect the application standing and does not determine acceptance.

## Evaluation Metric
Submissions are scored on the Root Mean Squared Error (RMSE), defined as:

$$ RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2} $$

where $\hat{y}_i$ is the predicted value and $y_i$ is the original value for each instance $i$.

## Submission File
For each id in the test set, the price of the car must be predicted. The file should contain a header and have the following format:

```
id,price
54273,39218.443
54274,39218.443
54275,39218.443
...
```

## Dataset Description
The dataset for this competition was generated from a deep learning model fine-tuned on the Used Car Price Prediction Dataset. Feature distributions are close to, but not exactly the same, as the original. Participants are free to use the original dataset as part of this competition.

## Files
- `train.csv` - the training dataset
- `test.csv` - the test dataset; the objective is to predict the value of the target Price
- `sample_submission.csv` - a sample submission file in the correct format


## Project Overview
This project is an endeavor to develop a predictive model for estimating car prices based on various vehicle attributes. The model is trained on a comprehensive dataset that includes information such as model year, manufacturer, mileage, and other pertinent features. The project employs Python libraries like Pandas and NumPy for data preprocessing, and PyTorch, a popular deep learning framework, for model development.

## Data Preprocessing
The preprocessing phase involves meticulous handling of missing data, encoding of categorical variables, and engineering of new features, such as the age of the car, calculated from the model year. This step ensures the data is in the right format for model training and potentially enhances the predictive capability of the model.

## Model Development
The model architecture comprises multiple fully connected layers with ReLU (Rectified Linear Unit) activation functions. Techniques such as dropout and batch normalization are incorporated to prevent overfitting and promote generalization, making the model more robust to variations in input data.

## Hyperparameter Optimization
The model undergoes a systematic hyperparameter tuning process to optimize its performance. This involves exploring different combinations of hyperparameters, such as the size of hidden layers and the dropout rate, using techniques like grid search and cross-validation. Each combination is evaluated on a validation set to determine the optimal set of hyperparameters.

## Model Evaluation
The performance of the model is evaluated using the mean squared error metric, which quantifies the discrepancy between predicted and actual prices. This evaluation provides insight into the model's accuracy and generalization ability, crucial for assessing its real-world applicability.

## Prediction and Submission
Finally, the trained model is deployed to make predictions on a separate test set. The predictions are then stored in a CSV file for submission.
## Tags
`#Python` `#Pandas` `#NumPy` `#PyTorch` `#DataPreprocessing` `#ModelDevelopment` `#HyperparameterOptimization` `#ModelEvaluation` `#KaggleX` `#SkillAssessmentChallenge` `#CarPricePrediction`
