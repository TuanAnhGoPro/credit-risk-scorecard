# Credit Risk Scorecard

This is my first data science portfolio project. I built a credit risk scorecard on Lending Club loan data, the same kind of model banks use to decide how risky a borrower is. I picked it because I want to work in banking and finance, and scorecards are still everywhere in that world.

## What I did

I started with the raw Lending Club file instead of a cleaned version from Kaggle, because I wanted to do the messy part myself. Then I cleaned the data, turned the variables into scores, tested the model on a later year of loans, and estimated how much money a bank could lose.

## The data

I used the accepted loans file from Lending Club (2007 to 2018). The data is too big for GitHub, so it is not in this repo. You can download it from Kaggle and put it in `data/raw/`.

Some choices I made on the way:

* I only kept loans where I know what happened in the end (paid back or charged off). Loans that were still running had no answer yet.
* I only used 36 month loans. A lot of the 60 month loans from later years had not finished by the end of the data, so their bad rate looked lower than it really is.
* I left out anything the bank would only know after the loan started, and also Lending Club's own grade and interest rate. Using them would make my model look better than it really is.
* I trained on loans from 2012 to 2014 (about 306 thousand loans, 13% went bad) and tested on 2015 (about 283 thousand loans, 15% went bad). I did it this way because a bank always scores new customers with a model built on older ones.

## How it works

First I cleaned the data and made a few new variables, like how many years of credit history someone has and their average FICO score. Some values made no sense, like an income of zero or a debt to income ratio above 100, so I set them to missing.

Then I used the `optbinning` package to group each variable into bins and calculate Weight of Evidence. Ten variables passed my Information Value filter. I fitted a logistic regression on top of that and turned the predicted probabilities into credit scores (600 points means 50 to 1 odds, and every 20 extra points doubles the odds).

After that I checked how stable the scores are, estimated expected credit loss as PD times LGD times EAD, and compared my model with gradient boosting.

## What I found

* The model scores AUC 0.654 on the 2015 test set (Gini 0.307, KS 0.221). On the training set it was 0.652, so it is not overfitting.
* The bad rate goes up steadily from about 4% in the safest group to about 28% in the riskiest group.
* The score distribution barely changed between train and test (PSI 0.001).
* LGD came out at about 0.89 and EAD at about 57% of the loan amount. Lending Club loans have no collateral, so losing almost everything on a default makes sense.
* The model predicted a loss rate of 6.37% for 2015 loans, but the real loss was 7.56%. So it was about 19% too optimistic.
* Gradient boosting got AUC 0.670, only 0.016 better. I would keep the scorecard because it is much easier to explain to a risk team.

## Things I noticed

The model underestimated risk for every group I checked in 2015. The customers looked the same as before, but more of them defaulted, so I think it was a general change in the market and not something about one type of borrower.

The variable `mort_acc` had a PSI of 0.20, which is quite high. When I looked into it, about 14% of the 2012 loans have no value for this variable, and that explains most of the shift. I decided it was a data problem and kept the missing group as it is.

The model also works worse for people with low income (AUC 0.61) than for people with high income (AUC 0.67).

## What is not perfect

* I did not discount the recoveries, so my expected loss is simpler than a proper IFRS 9 calculation.
* LGD and EAD are one number for the whole portfolio, not split by risk group.
* The PD covers the full 36 months, not just 12 months.
* The data has no gender, age or race, so I could not test fairness on protected groups. I only looked at income and home ownership, which are not the same thing.
* If I had more time, I would recalibrate the model on newer data and try more of the variables I did not use.

## Files

The notebooks are numbered in the order I made them: loading data, cleaning, WoE binning, modelling, scores and PSI, expected loss, and the SHAP and group checks. The charts are in `reports/figures`. To run it yourself, install the packages with `pip install -r requirements.txt` and run the notebooks in order.
