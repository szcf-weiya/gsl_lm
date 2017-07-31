# Poisson Regression

## Background

Poisson regression is useful when you're predicting an outcome variable representing counts from a set of continuous and/or categorical predictor variables.

I would use the dataset `breslow.dat` in the R package `robust` to illustrate how to implement poisson regression in R and cpp.

## Implement in R

In R, we can use the following command to implement poisson regression.

```
glm(, family = binomial())
```

More details refer to [rlogit.R](rlogit.R).

## Implement in GSL
