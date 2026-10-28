Synthèse des solutions des gagnants de la compétition Kaggle AMP®-Parkinson's Disease Progression Prediction

# Overview

Goal of the Competition
The goal of this competition is to predict MDS-UPDR scores, which measure progression in patients with Parkinson's disease. The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive assessment of both motor and non-motor symptoms associated with Parkinson's. You will develop a model trained on data of protein and peptide levels over time in subjects with Parkinson’s disease versus normal age-matched control subjects.

Your work could help provide important breakthrough information about which molecules change as Parkinson’s disease progresses.

Context
Parkinson’s disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions. Unfortunately, there is no current cure—and the disease worsens over time. It's estimated that by 2037, 1.6 million people in the U.S. will have Parkinson’s disease, at an economic cost approaching $80 billion. Research indicates that protein or peptide abnormalities play a key role in the onset and worsening of this disease. Gaining a better understanding of this—with the help of data science—could provide important clues for the development of new pharmacotherapies to slow the progression or cure Parkinson’s disease.

Current efforts have resulted in complex clinical and neurobiological data on over 10,000 subjects for broad sharing with the research community. A number of important findings have been published using this data, but clear biomarkers or cures are still lacking.

Competition host, the Accelerating Medicines Partnership® Parkinson’s Disease (AMP®PD), is a public-private partnership between government, industry, and nonprofits that is managed through the Foundation of the National Institutes of Health (FNIH). The Partnership created the AMP PD Knowledge Platform, which includes a deep molecular characterization and longitudinal clinical profiling of Parkinson’s disease patients, with the goal of identifying and validating diagnostic, prognostic, and/or disease progression biomarkers for Parkinson’s disease.

Your work could help in the search for a cure for Parkinson’s disease, which would alleviate the substantial suffering and medical care costs of patients with this disease.



This is a Code Competition. Refer to Code Requirements for details.

Evaluation
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.

For each patient visit where a protein/peptide sample was taken you will need to estimate both their UPDRS scores for that visit and predict their scores for any potential visits 6, 12, and 24 months later. Predictions for any visits that didn't ultimately take place are ignored.

You must submit to this competition using the provided python time-series API, which ensures that models do not peek forward in time. To use the API, follow this template in Kaggle Notebooks:

import amp_pd_peptide
env = amp_pd_peptide.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    sample_prediction_df['rating'] = np.arange(len(sample_prediction))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions

# 1

First of all we would like to thank Kaggle and AMP PD for hosting this competition and providing a great dataset to dig into and rich enough to get lost in it for several months. It is always an extra motivation to work on problems that can bring value to the medical field or contribute to scientific research. Additional words of gratitude of course go to @kyakovlev for the amazing work he has done, I think we formed a great team for this competition and the results reflect that.

We publish the full code of the winning notebook here

Quick summary
Our final solution is a simple average of two models: LGB and NN. Both models were trained on the same features (+ scaling/binarization for NN):

Visit month
Forecast horizon
Target prediction month
Indicator whether blood was taken during the visit
Supplementary dataset indicator
Indicators whether a patient visit occurred on 6th, 18th and 48th month
Count of number of previous “non-annual” visits (6th or 18th)
Index of the target (we pivot the dataset to have a single target column)
The winning solution fully ignores the results of the blood tests. We’ve tried hard to find any signal in this crucial piece of the data, but unfortunately we came to the conclusion that none of our approaches or models can benefit from blood test features significant enough to distinguish it from random variations. The final models were trained only on the union of clinical and supplementary datasets.

LGB
For the entire duration of the competition LGB was our model to beat and only a NN trained with the competition metric as the loss function was able to achieve competitive performance on CV. At first, we tried running a regression LGB model with different hyperparameters and custom objective functions, but nothing was better than l1 regression, which does not optimize the desired metric SMAPE+1. We also noticed that on CV the performance of every model is always better when the regression outputs are rounded to integers. Then we switched to an alternative approach.

Our LGB model is a classification model with 87 target classes (0 to maximum target value) and logloss objective. To produce the forecast we applied the following post-processing: given the predicted distribution of target classes, pick a value that minimizes SMAPE+1. Taking into account the observation that the optimal predictions are always integers, the task boils down to a trivial search among 87 possible values. Such an approach would have worked well for the original SMAPE metric also, because the approach treats cases with multiple local minimums naturally.

We ran an optimization routine to tune LGB hyperparameters to minimize SMAPE+1 on CV using the described post-processing.

NN
The neural network has a simple multi-layer feed-forward architecture with a regression target, using the competition metric SMAPE+1 as the loss function. We fixed the number of epochs and scheduler, and then tuned the learning rate and hidden layer size. The only trick there was to add a leaky relu activation as the last layer to prevent NN from getting stuck at negative predictions. Of course there are alternative ways to solve this issue.

Cross-Validation
We’ve tried multiple cross-validation schemes due to the small training sample size, all of them were stratified by patient id. Once a sufficient number of folds is used, they all are quite well correlated to each other. Better than to the public leaderboard :) The final scheme we relied on was leave-one-(patient)-out or, in other words, a group k-fold cross validation with a fold for each patient. We used it because it doesn’t depend on random numbers. The cross-validation correlated well enough with the private leaderboard, and the submit we chose turned out to be our best private LB submission.

What worked
The most impactful feature was the indication of whether a patient visit happened on the 6th month or not. It correlates strongly with the UPDRS targets (especially 2 and 3) and with frequency of medications being taken. As we can observe only the data correlation, it is impossible to judge what is the core reason for that. During the competition our hypothesis was that the patients that had more severe symptoms during the first examination (UPDRS scores parts 2 and 3) were more likely to get invited for a visit after 6 months and more likely to get medications prescribed. But for model training it was important that the patients that made a visit on the 6th month, have higher UPDRS scores on average. The same is true for an 18th month visit as well, but these 2 features are correlated. I personally wonder if presence / absence of these variables in the models are the reason for the private LB cliff effect around 20th place.

Another curious effect related to it is observed for the forecasts made at visit_month = 0. If you look at the model forecasts for 0, 12 and 24 months ahead, they are consistently lower than the forecasts 6 months ahead. It is very logical from the math point of view - if a patient will show up on the 6th month, they will have higher UPDRS scores on average, and if not - the forecast will be ignored. But such model behaviour is unreasonable from a clinical point of view of course.

It was also important to pay attention to the differences between training and test datasets as e.g. nicely summarized here. That, for instance, explains well why adding a feature indicating the visit on the 30th month could improve the CV, but ruin LB.

What didn’t work
Blood test data. We’ve tried many approaches to add proteins and/or peptides data to our models, but none of them improved CV. We narrowed it down to a bag of logistic regressions that forecasted the visit on the 6th month based on the blood test result on the 0th month. We applied soft up/down scaling of model-based predictions for patients that were more/less likely to visit on the 6th month based on the logistic regression probabilities. It worked on public LB after tuning a couple of “magic” coefficients directly on public LB itself. That gave us a boost all the way up to the second place on public LB, but it was clearly an overfit. We chose a “mild” version of that approach as our second final submission. It scored worse on private LB than the other submission, but, interestingly enough, not by as much as one could have expected (60.0 vs 60.3).

Thanks to everyone who participated in the competition, those who kept many interesting discussions going on the forum and those who suggested improvements! And congrats to all the winners!

# 2

orry for late sharing. First of all, thanks to competition organizers for hosting this interesting competition and my great teammates(@hyd,@rib). And congrats my friend rib for becoming GM.

Our solution is simple, we found information related to "visit_month" very useful and information related to proteins are useless. So we focus on the samples, structures and models.

Samples
We used samples with visit_month in "[0, 6, 12, 18, 24 , 36, 48, 60, 72, 84]" for training. And found that our cv is much better correlated to LB.

Features
All about “visit_month”
1) use visit_month as meta feature
2) gap between visit_month and last visit_month
3) times of visit for each patient
Little about “protein”
use NPX’s ratio of each patient, instead of the original values

Structure
16 labels
Predict each “updrs” for predicted_month_diff=0,6,12,24

4 labels
Use predicted_month_diff as a feature, and predict each one’s 4 updrs

Models
Simple MLP with different structures and parameters.Finally we blends with:

Models training in different numbers of labels
Models training in different structure of network
Models training in different parameters of network


# 3

First of all, I would like to thank Kaggle and AMP PD for organizing this great competition. I would also like to thank all the kagglers who participated in this competition. I am very happy to have won my first gold medal and also to have won a prize.

My solution is simple and consists of three main functions.

1. Grouping
As many have pointed out, I have divided the groups into two, each optimized for a different visit interval (6 or 12 months).
One key point is that we need to focus not only on the 6th month, but also on the 18th month. There are patients who are missing the 6th month but have the 18th month present, and this patient is not healthy.
By using the cumulative minimum function, I considered patients with either the 6th month or the 18th month present as unhealthy.

[Group A : Healthy]
Patients with a minimum visit interval of 12 months or more
[Group B : Unhealthy]
Patients with a minimum visit interval of 6 months or less
2. Labeling (mainly Group B)
As I looked at the data for Group B, the unhealthy group, I found several patterns. The frequency of protein collection and protein information are linked to the severity of symptoms. I then generated several labels based on protein collection frequency and protein information and used them as features.

The following 9 labels were finally adopted.

[more severe symptoms]
Protein was collected at 6 months
Protein was collected at 6 months and again at 12 months
Low number of unique "UniPort" (approximately the bottom 20%)
Low number of unique "UniPort" (approximately the bottom 10%)
[milder symptoms]
Protein not collected at 6 months
Protein was collected at 6 months but not at 12 months
Protein not collected at 6 months, but collected at 18 months
High number of unique "UniPort" and high change of "Peptide" from the previous measurement (approximately the top 20%)
High number of unique "UniPort" and high change of "Peptide" from the previous measurement (approximately the top 10%)
3. Modeling
Initially, I also tried LightGBM using these features, but LB worsened and I did not use them in my final model. In my final model, I used these features to obtain the coefficients (severity) by grid search.
Due to the small sample size of the train data (248 patients), some labels improved for the train data (248 patients) but worsened for the LB (50 patients). In my various experiments, I decided to adopt only those that improved both train and LB (248 + 50 patients).
I thought this would be a more robust model. As a result, the final scores were also stable.