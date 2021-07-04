# HOSA
Subgroup analysis via Alternating Direction Method of Multipliers (ADMM) algorithm and statistical inference on subgroup effects via Expectation-Maximization (EM) algorithm

## Installation
```R
### install devtools packages (devtools package) if not installed
> install.packages("devtools")

### install HOSA package
> devtools::install_github("sqsun/HOSA")
> library(HOSA)
```

## An example -- HOSA package
```R
### generate an example data to testing
> num_sample <- 200
> isig <- 1
> il <- 1
> irpt <- 1
> SIGMA2B = list(c(0.1,0.1,0.1),c(0.5,0.5,0.5),c(1,1,1),c(2,2,2),c(4,4,4))
> em_iter = 1000
> library(MASS)
> sigma2b = SIGMA2B[[isig]]
> set.seed(irpt)
> sigma2e = 1
> alpha = 0.4                 # coefficient

#### generate X and W
> q0 = 1
> coe = 0
> Sigma = matrix(coe, q0+1, q0+1)
> diag(Sigma) = 1
> xw = mvrnorm(num_sample, rep(0, q0+1), Sigma)
> X = xw[,1]
> W = xw[, -1]

#### generate pi
> K = 3
> q2 = 2
> Eta2 = matrix(rep(c(-1, 1, 0), q2), K, byrow = FALSE)
> Eta2 = matrix(c(-1, -1, 0, 1, -1, 0), 3)
> Z2 <- t( matrix(cbind(rep(1,num_sample), rnorm(num_sample)), ncol=q2) )

> bas = exp(Eta2 %*% Z2)
> pi = bas/matrix(rep(apply(bas, 2, sum),K), K, byrow = TRUE)

> q1 = 2
> Eta1 = matrix(rep(c(-1, 1, 3), q1), K, byrow = FALSE)
> Z1 <- t( matrix(cbind(rep(1,num_sample),rnorm(num_sample)),ncol=q1) )

> mu = Eta1 %*% Z1

> random_sample <- beta <- rep(NA, num_sample)
> for(i in 1:num_sample){
>   sid = sample(c(1:3), 1, prob=pi[,i])
>   random_sample[i] = sid
>   beta[i] = rnorm(1, mu[sid,i], sqrt(sigma2b[sid])) 
> }#end for

> epsl = rnorm(num_sample, 0, sqrt(sigma2e) )
> y = X*beta + W*alpha + epsl

### run HOSA
> num_sample = length(y)
> LAMBDA = seq(0.01, 1, by = 0.1)
> il = 1
> lambda = LAMBDA[il]
> K = 3
> init_Eta1 <- matrix(rnorm(num_sample*q1), ncol = q1)

> admm_iter <- 1000
> em_iter = 1000
>tol <- 1e-3
> res_path <- "./"

> res_admm <- subG_ADMM_extension(y, X, as.matrix(W), t(Z1), init_Eta1, lambda, admm_iter, tol)

> rk <- kmeans(res_admm$Eta1, centers=K, nstart=10000, iter.max=5000)

> pi_init = matrix(rep(1/K, K*num_sample), num_sample)
> Eta2_init = matrix(0, q2, K)
> res_em <- EMstep_extension(y, X, Z1, Z2, t(as.matrix(W)), 1, rep(1, K), pi_init, t(rk$centers), Eta2_init, res_admm$alpha, em_iter)
> res_em$rk <- rk

### save the results
> save(res_em, file=paste0(res_path,"/res.initRandom.ss",num_sample,".lambda",lambda,".RData") )

```
## Citation
Ling Zhou, Shiquan Sun, Haoda Fu and Peter X.K. Song. *Subgroup-effects models for the analysis of personal treatment effects*. Annals of Applied Statistics, 2021, in press 
