#' Subgroup Analysis via ADMM and EM
#' extract the reference
#' @param x a data matrix 
#' @param xt a data matrix
#' @return a list.
#' @export
Ref <- function(x, xt){
	j = ncol(x)
	nx = matrix(NA, nrow(x), j)
	nlab = rep(NA, j)
	for(jj in 1:j){
		lab = which.min(apply((x - outer(xt[, jj], rep(1, j)))^2,2,sum))
		nx[, jj] = x[, lab]
		nlab[jj] = lab
	}# end for
	return(list(nx = nx, nlab = nlab))
}# end func
