Lab 9 - HPC
================

# Learning goals

In this lab, you are expected to practice the following skills:

- Evaluate whether a problem can be parallelized or not.
- Practice with the parallel package.
- Use Rscript to submit jobs.

## Problem 1

Give yourself a few minutes to think about what you learned about
parallelization. List three examples of problems that you believe may be
solved using parallel computing, and check for packages on the HPC CRAN
task view that may be related to it.

- cross-validation in machine learning

- carnet -\> supports parallel cross-validation with `doParallel`

- mlr, foreach, doParallel -\> for parallel model traning

- bootstrapping

- boot -\> for bootstrapping

- parallel -\> parallelize resampling

- markov chain monte carlo

- parallel

- rstan -\> for stan for bayesian modeling

- RcppParallel -\> parallel mcmc sampling

- nimle -\> customize bayesian inference

## Problem 2: Pre-parallelization

The following functions can be written to be more efficient without
using `parallel`:

1.  This function generates a `n x k` dataset with all its entries
    having a Poisson distribution with mean `lambda`.

``` r
fun1 <- function(n = 100, k = 4, lambda = 4) {
  x <- NULL
  
  for (i in 1:n)
    x <- rbind(x, rpois(k, lambda))
  
  return(x)
}

fun1alt <- function(n = 100, k = 4, lambda = 4) {
  matrix(rpois(n * k, lambda), nrow = n, ncol = k)
}

# Benchmarking
microbenchmark::microbenchmark(
  fun1(100),
  fun1alt(100),
  unit = "ns"
)
```

    ## Unit: nanoseconds
    ##          expr    min     lq     mean   median       uq      max neval
    ##     fun1(100) 247925 292300 387879.4 357899.5 393875.0  3758862   100
    ##  fun1alt(100)  16806  18723 157355.4  19503.5  22655.5 13419216   100

How much faster?

*fun1(100) is about 7 times faster than fun1alt(100) by comparing with
mean of them.*

2.  Find the column max (hint: Checkout the function `max.col()`).

``` r
# Data Generating Process (10 x 10,000 matrix)
set.seed(1234)
x <- matrix(rnorm(1e4), nrow=10)

# Find each column's max value
fun2 <- function(x) {
  apply(x, 2, max)
}

fun2alt <- function(x) {
  x[cbind(max.col(t(x)),1:ncol(x))]
}

# Benchmarking
bench <- microbenchmark::microbenchmark(
  fun2(x),
  fun2alt(x),
  unit = "us"
)
bench
```

    ## Unit: microseconds
    ##        expr      min       lq      mean   median        uq      max neval
    ##     fun2(x) 1079.300 1161.560 1339.1669 1199.119 1278.2325 4269.028   100
    ##  fun2alt(x)  106.101  142.063  188.9465  157.847  180.5815 2603.452   100

``` r
plot(bench)
```

![](lab09-hpc_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
ggplot2::autoplot(bench) +
  ggplot2::theme_minimal()
```

![](lab09-hpc_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

- The first boxplot shows that fun2(x) has higher execution times with
  higher outliers, while fun2alt(x) exhibits lowerexecution times with
  lower outliers. The second violin plot confirms that fun2alt(x) not
  only has a lower median but also a much more concentrated density,
  which is better.

## Problem 3: Parallelize everything

We will now turn our attention to non-parametric
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
Among its many uses, non-parametric bootstrapping allow us to obtain
confidence intervals for parameter estimates without relying on
parametric assumptions.

The main assumption is that we can approximate many experiments by
resampling observations from our original dataset, which reflects the
population.

This function implements the non-parametric bootstrap:

``` r
library(parallel)
my_boot <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)
 
  # Making the cluster using `ncpus`
  # STEP 1: GOES HERE
  cl <- makePSOCKcluster(ncpus)
  # STEP 2: GOES HERE
  clusterExport(cl, varlist = c("idx", "dat", "stat"), envir= environment())
  
  # STEP 3: THIS FUNCTION NEEDS TO BE REPLACED WITH parLapply
  ans <- parLapply(cl, seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  # STEP 4: GOES HERE
  stopCluster
  
  ans
  
}
```

1.  Use the previous pseudocode, and make it work with `parallel`. Here
    is just an example for you to try:

``` r
# Bootstrap of a linear regression model
my_stat <- function(d) coef(lm(y~x, data =d))

# DATA SIM
set.seed(1)
n <- 500 
R <- 1e4
x <- cbind(rnorm(n)) 
y <- x*5 + rnorm(n)

# Check if we get something similar as lm
ans0 <- confint(lm(y~x))
cat("OLS CI \n")
```

    ## OLS CI

``` r
print(ans0)
```

    ##                  2.5 %     97.5 %
    ## (Intercept) -0.1379033 0.04797344
    ## x            4.8650100 5.04883353

``` r
ans1 <- my_boot(dat = data.frame(x,y), my_stat, R=R, ncpus = 4)
qs <- c(.025, .975)
cat("Bootstrp CI \n")
```

    ## Bootstrp CI

``` r
print(t(apply(ans1, 2, quantile, probs = qs)))
```

    ##                   2.5%      97.5%
    ## (Intercept) -0.1386903 0.04856752
    ## x            4.8685162 5.04351239

2.  Check whether your version actually goes faster than the
    non-parallel version:

``` r
parallel::detectCores()
```

    ## [1] 8

``` r
system.time(my_boot(dat = data.frame(x,y), my_stat, R=4000, ncpus = 1L))
```

    ##    user  system elapsed 
    ##   0.075   0.016   4.170

``` r
system.time(my_boot(dat = data.frame(x,y), my_stat, R=4000, ncpus = 8L))
```

    ##    user  system elapsed 
    ##   0.204   0.063   1.828

*ncpus = 8L is slightly faster than ncpus = 1L*

## Problem 4: Compile this markdown document using Rscript

Once you have saved this Rmd file, try running the following command in
your terminal:

``` bash
Rscript --vanilla -e 'rmarkdown::render("~/Downloads/lab09-hpc.Rmd")'
```

Where `[full-path-to-your-Rmd-file.Rmd]` should be replace with the full
path to your Rmd file… :).
