# Thesis Repository
This repository serves as the location where I save my thesis files and progress. 
It'll also serve as backup for my files.

## Thesis Defense Feedback (CMSC 199.1):
1. The methods ﬁx the hyperparameters for all models. Should explore if by hard-coding these parameters, a major source of the instability is not intentionally excluded, which defeats the problem statement. Explore if optimization of hyperparameter values will lead to better results.
2. The problem statement distinguishes between dataset instability and stochastic instability. How do you ensure that these two distinct sources are isolated and evaluated independently, rather than just observing their combined eﬀect?
3. Further discussion on how to prevent the inherent randomness of ensemble models from
confusing measurements when trying to isolate stochastic vs dataset instability (speciﬁcally Random Forest's internal bootstrap sampling and random feature selection)
4. Add seed for reproducibility. 
5. Clearly deﬁne and formalize instability metrics; ensure reproducibility across runs and datasets.
6. Maybe add baseline comparisons with standard fairness/metrics to contextualize the result
7. Can a model be fair but unstable or stable but fair?
8. Do you think bootstrap resampling is enough for your case to study instability? Why?"
9. Best to provide preliminary results in your paper.
10. Determine clearly the instability that aﬀects the result.