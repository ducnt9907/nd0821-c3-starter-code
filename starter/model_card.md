# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a **Random Forest Classifier** developed for binary classification, predicting whether an individual's income exceeds $50K per year based on demographic features. The model uses various categorical and numerical features from a dataset, including attributes like age, work class, education, marital status, occupation, race, sex, and hours worked per week.

### Features Used
- **Categorical Features**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Numerical Features**: age, fnlgt (final weight), education-num, capital-gain, capital-loss, hours-per-week

## Intended Use
The model is intended for exploratory data analysis or educational purposes to demonstrate machine learning workflows in classification tasks involving demographic data. It could be used to understand feature importance in income prediction but is not suitable for deployment in sensitive applications without further refinement and ethical review.

## Training Data
The training data consists of a subset of data from the **U.S. Census**, covering demographic information and income categories. The categorical and numerical features include data points such as education level, occupation, race, and capital gains/losses. The labels (target variable) indicate whether the individual's income is `<=50K` or `>50K`.

## Evaluation Data
The evaluation data is derived from a similar demographic dataset, preprocessed to match the training data's categorical encoding and label binarization. The dataset was split into training and test sets using an 80-20 split. This means 80% of the data was used for training, and 20% was reserved for evaluation. The test set includes diverse samples from the same feature distribution as the training data to ensure consistency in evaluation.

## Metrics
- **Precision, Recall, Fbeta**: Calculated for each class to evaluate performance on slices of the data.
- **Slice-based Evaluation**: Model performance is further analyzed across slices based on categorical features (e.g., `workclass`, `education`, `race`, etc.), enabling insights into how the model performs on specific subgroups.

## Ethical Considerations
This model uses demographic data that includes sensitive attributes such as race, sex, and marital status. Such attributes can lead to biased predictions if the model's decisions are used in real-world applications without bias mitigation. Given the potential for reinforcement of stereotypes and the historical bias present in demographic datasets, this model should be used cautiously, especially in decision-making that impacts individuals' socioeconomic opportunities.

## Caveats and Recommendations
- **Data Bias**: The training data may contain biases reflecting societal inequalities. This could lead to unfair predictions if the model is deployed without proper adjustments.
- **Limitations of Random Forest**: The model may not generalize well to new data outside the specific feature distribution it was trained on.
- **Not Suitable for Deployment**: Due to the ethical concerns and potential biases in demographic features, this model is not recommended for deployment in any decision-making process affecting individuals.
