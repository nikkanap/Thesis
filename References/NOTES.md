# Reference Notes

## Main References

## References Read
1. Black-box tests for algorithmic stability
- The article talks about statistical ways to measure stability in algorithms. 
- Can be used as steps for measuring instability in the experiment (methodology reference)
2. Forks over Knives
- Talks about different sources that may cause prediction inconsistency 
- Can be used to directly reference stochastic instability
- Can be used to reference things like fairness and other biases that can be avoided in the thesis
3. Stability of Machine Learning Algorithms
- Talks about Decision Boundary Instability and Classification Instability
- We can use DBI as an extension and reference our method as a focus on CI only.
- Also has lots of definitions on instability.
4. Evaluating Model Robustness and Stability to Dataset Shift
- Offers the idea of stress testing models since models are usually tested with data similar to the training data
- stress testing using the weakest (the one the model performs poorly in) data from the existing data (training data)
5. Prediction Instability in Machine Learning Ensembles
- Talks about ensembles and how prediction inconsistency is an intrinsic part of it like random forest and xgboost
- doesn't feel as significant of a reference honestly so review this in the paper (except for referencing random forest and xgboost ig)
6. Stability and Generalization of Learning Algorithms that Converge to Clobal Optima
- mostly talks about stochastic gradients and methods to train models for black box models like neural networks
- not really useful in our case (as of now)
- can be used to reference some things but not really necessary
7. Underspecification presents challenges for credibility in model machine learning
- similar to the article `(4)` where they do stress testing  but with the focus on underspecification 
- can be used as reference for "changing the random seed" for stochastic instability
8. Recidivism Forecasting Using XGBoost
- uses Brier Score as performance metric (aligns with our study)
- shows the excellent performance of xgboost with recidivism risk assessment
9. An open source replication of a winning recidivism prediction model
- winning solution to the NIJ competition
- models were Lasso Logistic Regression and XGBoost (they performed eequivalently to each other)
- Lasso LR is more readable than XGBoost
10. Risk-need-responsivity model for offender assessment and rehabilitation
- mostly talks about RNR model, how it influence assessment tool, and how to address criminal needs and such
- mostly just for referencing surface level stuff like the history of recidivism risk assessment models
11. Out with the old and in with the new? An empirical comparison of supervised learning algorithms to predict recidivism
- a lot of good content talking about the history of ML in recidivism prediction and even how they came to be
- compared most of the models we're testing for performance
- maybe we can replace SVM with ANN or something? (the article mentions testing ANN)
12. 

