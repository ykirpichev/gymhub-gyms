# A naive Bayes posterior

A spam filter is a naive Bayes model with two classes and equal priors, `P(spam) = P(ham) = 0.5`. Given the class, the two binary features A and B are independent. You observe `A = 1`, `B = 1`, and

| | P(feature = 1 \| spam) | P(feature = 1 \| ham) |
|---|---|---|
| A | 0.8 | 0.4 |
| B | 0.6 | 0.3 |

What is `P(spam | A = 1, B = 1)`? Give at least two decimal places.
