#include <fstream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <R.h>
#include <Rmath.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <ctime>
#include <Rcpp.h>

//#include "polyroot.h"

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace std;
using namespace arma;
using namespace Rcpp;

#define ARMA_DONT_PRINT_ERRORS

//
//**********************************************************************//
//                   Fitting Logistic Regression                        //
//**********************************************************************//

//' fitting multivariate logistic regression
//' @param Gamma -- dim2 x num_k
//' @param Z2 -- num_sample x dim2
//' @param Eta2 -- dim2 x num_k
//' 
//' 
void MLogisticRegression(arma::mat Z2, arma::mat Gamma, arma::mat &Eta2){
   
    //cout<<"compute pi"<<endl;
    const int num_k = Gamma.n_cols;
    const int num_var = Eta2.n_rows;
    const int num_sample = Gamma.n_rows;

    const int nr_iter = 100; // maximum iteration of newton raphson

    // pre-define score and information matrix are from first derivative and second deriviative, respectively.
    arma::mat JM = zeros<arma::mat>(num_var * (num_k - 1), num_var * (num_k - 1));
    arma::mat J = zeros<arma::mat>(num_var, num_var);
    arma::vec score = zeros<arma::vec>(num_var * (num_k - 1));

    // parameter to be estimated, the column num_k - 1 is zeros
    arma::vec tau = vectorise(Eta2.cols(0, num_k - 2));
    arma::vec tau_old = tau;
    int true_iter = 1;
    while (true_iter)
    {
        // Z2Eta2 -- num_dim2 x num_k
        arma::mat Z2Eta2 = Z2.t() * Eta2;
        arma::vec denom = zeros<arma::vec>(num_sample);
        for (size_t k = 0; k < (num_k - 1); k++)
        {
            denom += exp(Z2Eta2.col(k));
        } // end for

        // compute Jaccobi matrix, second derivatives
        for (size_t k1 = 0; k1 < (num_k - 1); k1++)
        {
            arma::vec pi1 = exp(Z2Eta2.col(k1)) / (1.0 + denom);
            for (size_t k2 = k1; k2 < (num_k - 1); k2++)
            {
                if (k1 == k2)
                {
                    J = Z2 * diagmat((1 - pi1) % pi1) * Z2.t();
                    JM.submat(num_var * k1, num_var * k2, num_var * (k1 + 1) - 1, num_var * (k2 + 1) - 1) = J;
                }
                else
                {
                    arma::vec pi2 = exp(Z2Eta2.col(k2)) / (1.0 + denom);
                    // Date:2019-1-18 08:23:03
                    //J = Z2 * diagmat(pi1 % pi2) * Z2.t();
                    J = - Z2 * diagmat(pi1 % pi2) * Z2.t();
                    JM.submat(num_var * k1, num_var * k2, num_var * (k1 + 1) - 1, num_var * (k2 + 1) - 1) = J;
                    JM.submat(num_var * k2, num_var * k1, num_var * (k2 + 1) - 1, num_var * (k1 + 1) - 1) = J;
                } // end fi

            } // end for
            // compute first derivatives
            score.subvec(num_var * k1, num_var * (k1 + 1) - 1) = Z2 * (Gamma.col(k1) - pi1);
        } // end for

        // *update tau, the order is small, equals to the number of Aations
        tau = tau_old + solve(JM, score);

        //cout<<"Mstep:: tau = "<<tau<<endl;
        if (norm(tau_old - tau) < 1e-03 || (++true_iter) > nr_iter)
        {
            break;
        }
        else
        {
            tau_old = tau;
            Eta2.cols(0, num_k - 2) = reshape(tau, num_var, num_k - 1);
        } // end fi
    }     // end while
    Eta2.cols(0, num_k - 2) = reshape(tau, num_var, num_k - 1);
} // end func

//**********************************************************//
//                   Expectation Step                       //
//**********************************************************//
//' Expectation
//'
//' @param y -- num_sample x 1
//' @param X -- num_sample x 1
//' @param Z1 -- num_dim1 x num_sample
//' @param Eta1 -- num_dim1 x num_k
//'
void Estep_extension(arma::vec y, arma::vec X, arma::mat Z1, arma::mat Z2, arma::mat W, arma::mat &A, arma::mat &B, arma::mat &Gamma, arma::vec sigma2b, double sigma2e, arma::mat pi, arma::mat Eta1, arma::vec alpha) {
    // *compute A, and B
    const int num_k = sigma2b.n_elem; // number of clusters
    const int num_sample = y.n_elem;  // number of samples
    for (size_t k = 0; k < num_k; k++)
    { // A -- num_sample x num_k;  B -- num_sample x num_k;
        //A.col(k) = (X % X) / sigma2e + sigma2b(k);
        //Date:2019-1-18 08:24:18
        A.col(k) = (X % X) / sigma2e + 1.0/sigma2b(k);
        B.col(k) = ((y - W.t() * alpha - X % (Z1.t() * Eta1.col(k))) % X) / sigma2e;
    } // end for
    
	//cout << "Update AB" << endl;
	
    // *compute M matrix -- num_sample x num_k
    arma::mat M = zeros<arma::mat>(num_sample, num_k);
    arma::vec Walpha = W.t() * alpha;
    for (size_t i = 0; i < num_sample; i++)
    {
        for (size_t k = 0; k < num_k; k++)
        {
            double residual = y(i) - Walpha(i) - X(i) * dot(Z1.col(i), Eta1.col(k));
            double norm_const = 2.0 * 3.141593 * sigma2e * sigma2b(k) * abs(A(i, k));
            M(i, k) = 1.0 / sqrt(norm_const) * exp(0.5 * (B(i, k) * B(i, k)) / A(i, k) - 0.5 * residual * residual / sigma2e);
        } // end for
    }     // end for

	//cout<<"Update M"<<endl;
	
    Gamma = M % pi;         //midification by Ling Zhou//
    // normalized Gamma to make sure summation of each row is equal to 1
    for (size_t i = 0; i < num_sample; i++)
    {
        Gamma.row(i) = Gamma.row(i) / sum(Gamma.row(i));
    } // end for
	
	//cout<<"Update Gamma"<<endl;

} // end Estep

//************************************************************//
//                   Maximization Step                        //
//************************************************************//
//' Maximization
//'
void Mstep_extension(arma::vec y, arma::vec X, arma::mat Z1, arma::mat Z2, arma::mat W, arma::mat A, arma::mat B, arma::mat Gamma, double &sigma2e, arma::vec &sigma2b, arma::mat &Eta1, arma::mat &Eta2, arma::mat &pi, arma::vec &alpha){
    // * M step
    const int num_k = sigma2b.n_elem;
    const int num_sample = y.n_elem;
    const int num_dim1 = Z1.n_rows;
    const int num_dim2 = Z2.n_rows;
    //======================================
    // *update pi -- num_sample x num_k
    //cout<<"Update pi"<<endl;
    //arma::mat pi = zeros<arma::mat>(num_sample, num_k);
    arma::mat Z2Eta2 = Z2.t() * Eta2; // num_dim1 x num_k
    arma::vec denom_vec = zeros<arma::vec>(num_sample);
    for (size_t k = 0; k < (num_k - 1); k++)
    {
        denom_vec += Z2Eta2.col(k);
    } // end for
    for (size_t k = 0; k < (num_k - 1); k++)
    {
        pi.col(k) = exp(Z2Eta2.col(k)) / (1.0 + denom_vec);
    } // end for
    pi.col(num_k - 1) = 1.0 / (1.0 + denom_vec);

    //======================================
    // *update Eta1 -- num_dim1 x num_k
    //cout<<"Update Eta1"<<endl;
    arma::vec Walpha = W.t() * alpha;
    for (size_t k = 0; k < num_k; k++)
    {
        arma::mat denom = zeros<arma::mat>(num_dim1, num_dim1);
        for (size_t i = 0; i < num_sample; i++)
        {
            arma::vec tmp = Z1.col(i) * X(i);
            denom += Gamma(i, k) * tmp * tmp.t();
        } // end for i
        arma::vec numer = zeros<arma::vec>(num_dim1);
        for (size_t i = 0; i < num_sample; i++)
        {
            double residual = y(i) - Walpha(i) - X(i) / A(i, k) * B(i, k);
            numer += Gamma(i, k) * residual * X(i) * Z1.col(i);
        } // end for i
        Eta1.col(k) = solve(denom, numer);
    } // end num_k
    //======================================
    // *update Eta2 -- dim2 x k, the last column of Eta2 is zero
    //cout<<"Update Eta2"<<endl;
    // using multinomial logistic regression
    MLogisticRegression(Z2, Gamma, Eta2);

    //=======================================
    // *update alpha -- c x 1
    //cout<<"Update alpha"<<endl;
    arma::mat WtW = W * W.t(); // W -- c x c
    arma::vec ssr_alpha = zeros<arma::vec>(num_sample);
    for (size_t i = 0; i < num_sample; i++)
    {
        for (size_t k = 0; k < num_k; k++)
        { // accsum all clusters
            ssr_alpha(i) += Gamma(i, k) * (y(i) - X(i) * dot(Z1.col(i), Eta1.col(k)) - X(i) / A(i, k)* B(i, k));
        } // end for
    }     // end for
    //alpha = inv_sympd(WtW)*as_scalar( accu(ssr_alpha) );
    alpha = inv_sympd(WtW) * W * ssr_alpha;

    //=================================================
    // *update sigma2e -- 1 x 1, scaler
    //cout<<"Update sigma2e"<<endl;
    double ssr_sigma2e = 0;
    double residual = 0.0;
    Walpha = W.t() * alpha; // the updated alpha to do the next step
    for (size_t i = 0; i < num_sample; i++)
    {
        for (size_t k = 0; k < num_k; k++)
        {
            residual = y(i) - Walpha(i) - X(i) * dot(Z1.col(i), Eta1.col(k));
            ssr_sigma2e += Gamma(i, k) * ( (X(i) / A(i, k)) * (1.0 + B(i, k) * B(i, k) / A(i, k)) * X(i) -
                           2 * residual * X(i) * B(i, k) / A(i, k) +
                           residual * residual );
        } // end for
    }     // end for
    sigma2e = sum(ssr_sigma2e) / (double)num_sample;

    //=====================================================
    // *update sigma2b -- num_k x 1
    //cout<<"Update Sigma"<<endl;
    for (size_t k = 0; k < num_k; k++)
    {
        double ssr_sigma = 0;
        for (size_t i = 0; i < num_sample; i++)
        {
            ssr_sigma += Gamma(i, k) * (1.0 + B(i, k) * B(i, k) / A(i, k)) / A(i, k);
        } // end for i
        sigma2b(k) = ssr_sigma / sum(Gamma.col(k));
    } // end for k
    //	return 0;
} // end function

//*******************************************************************//
//                EXPECTATION MAXIMIZATION                           //
//*******************************************************************//
//' modified by sun
//' date: 2018-12-30 09:44:04
//' EM algorithm to fit the model
//'
//' @param y -- num_sample x 1
//' @param X -- num_sample x 1
//' @param Z1 -- num_dim1 x num_sample
//' @param Eta1 -- num_dim1 x num_k
//'
//' @return A list
//' 
//' @export
// [[Rcpp::export]]
SEXP EMstep_extension(SEXP yin, SEXP Xin, SEXP Z1in, SEXP Z2in, SEXP Win, SEXP sigma2ein, SEXP sigma2bin, SEXP piin, SEXP Eta1in, SEXP Eta2in, SEXP alphain, SEXP em_iterin){
    try
    {                           // *EM Algorithm
        arma::vec y = as<arma::vec>(yin);   // *dim = num_sample x 1
        arma::vec X = as<arma::vec>(Xin);   // *dim = num_sample x 1
        arma::mat Z1 = as<arma::mat>(Z1in); // *dim = num_dim x num_sample, expert knowledge
        arma::mat Z2 = as<arma::mat>(Z2in); // *dim = num_dim x num_sample, expert knowledge
        //arma::mat Z2 = Z2_tmp.t();
        
        arma::mat W = as<arma::mat>(Win);   // *dim = c x num_sample, covariates

        
        arma::vec alpha = as<arma::vec>(alphain);
        arma::mat pi = as<arma::mat>(piin);

        // *initial values
        double sigma2e = as<double>(sigma2ein); // epsilon
        //const Rcpp::List PHI(Sigmain);
        arma::vec sigma2b = as<arma::vec>(sigma2bin);

        const int num_dim1 = Z1.n_rows;
        const int num_dim2 = Z2.n_rows;
        const int num_sample = y.n_elem;
        const int num_k = sigma2b.n_elem;
        const int em_iter = Rcpp::as<int>(em_iterin);

        // *definition
        //arma::mat Eta1 = zeros<arma::mat>(num_dim1, num_k);
        arma::mat Eta1 = as<arma::mat>(Eta1in);
        //arma::mat Eta2 = zeros<arma::mat>(num_dim2, num_k);
        arma::mat Eta2 = as<arma::mat>(Eta2in);
        arma::mat A = zeros<arma::mat>(num_sample, num_k);
        arma::mat B = zeros<arma::mat>(num_sample, num_k);
        arma::mat Gamma = zeros<arma::mat>(num_sample, num_k);
        arma::mat Eta1_old = Eta1;
        arma::mat Eta2_old = Eta2;
        size_t iter = 0;
        for (iter = 0; iter < em_iter; iter++)
        {
            // *E-step, find Gamma
            cout << "E-step: " << iter + 1 << endl;
            double tstart1 = clock();
            // Estep_extension(arma::vec y, arma::vec X, arma::mat Z1, arma::mat Z2, arma::mat W, arma::mat &A, arma::mat &B, arma::mat &Gamma, arma::vec sigma2b, double sigma2e, arma::mat pi, arma::mat Eta1, arma::vec alpha)
            Estep_extension(y, X, Z1, Z2, W, A, B, Gamma, sigma2b, sigma2e, pi, Eta1, alpha);
            double time_mcmc = (clock() - tstart1) / (double(CLOCKS_PER_SEC));

            // * M-step, find sigmae2, sigma2, pi, alpha
            cout << "M-step: " << iter + 1 << endl;
            double tstart2 = clock();
            Mstep_extension(y, X, Z1, Z2, W, A, B, Gamma, sigma2e, sigma2b, Eta1, Eta2, pi, alpha);
            double time_nr = (clock() - tstart2) / (double(CLOCKS_PER_SEC));
            if (max(as_scalar(accu(abs(Eta1_old - Eta1))) / as_scalar(accu(abs(Eta1))), as_scalar(accu(abs(Eta2_old - Eta2))) / as_scalar(accu(abs(Eta2)))) < 1e-3)
            {
                break;
            }
            else
            {
                Eta1_old = Eta1;
                Eta2_old = Eta2;
            } // end fi
        }     // end for em_iter
        return List::create(Named("sigma2e") = sigma2e, Named("sigma2b") = sigma2b, Named("Eta1") = Eta1, Named("Eta2") = Eta2, Named("alpha") = alpha, Named("pi") = pi, Named("Gamma") = Gamma, Named("num_iter") = iter-1);
    }
    catch (std::exception &ex)
    {
        forward_exception_to_r(ex);
    }
    catch (...)
    {
        ::Rf_error("C++ exception (unknown reason)...");
    }
    return R_NilValue;
} // end func

////////////////////////////////////////////////////////////////////////////////
// date: 2019-1-5 11:19:23
// modified by sun
arma::mat KronProd(arma::mat A, arma::mat B)
{
    return (kron(A, B));
} // end func

// row pair-wise product
arma::mat RowWiseKronProd(arma::mat A, arma::mat B)
{
    const int num_row = A.n_rows;
    const int num_colA = A.n_cols;
    const int num_colB = B.n_cols;
    arma::mat res_kron = zeros<arma::mat>(num_row, num_colA * num_colB);
    for (size_t i = 0; i < num_row; i++)
    {
        res_kron.row(i) = kron(A.row(i), B.row(i));
    } // end for
    return (res_kron);
} // end func

void InitializeEta1(arma::mat Z, arma::vec Y, arma::mat IsQ, arma::mat VtV, arma::mat &Eta1)
{
    const int max_iter = 500;
    const int num_var = Z.n_cols;
    const int num_sample = Z.n_rows;
    arma::mat Eta1_old = Eta1;
    int iter = 1;
	double aa = 0.0;
    while (iter < max_iter)
    {
        arma::vec Y_tmp = zeros<arma::vec>(num_sample);
        for (size_t i = 0; i < num_var; i++)
        {
            Y_tmp += Z.col(i) % Eta1.col(i);
        } // end for

		//cout << "Y_tmp" << endl; 
		
        for (size_t i = 0; i < num_var; i++)
        {
            arma::vec Y_iter = Y - ( Y_tmp - Z.col(i) % Eta1.col(i) );
			//cout << "Y_iter" << endl;
            arma::mat Zdiag = diagmat(Z.col(i));
			//cout << "Zdiag" << endl;
            Eta1.col(i) = inv(Zdiag.t() * IsQ * Zdiag + 0.001 * VtV) * (Zdiag * IsQ * Y_iter);
			//cout << "Eta1.col" << endl;
        } // end for
		
		//cout << "Eta1.col" << endl;

        // to check stop criteria
        if (norm(Eta1_old - Eta1, 2) < 1e-3)
        {
            aa = norm(Eta1_old - Eta1, 2);
			cout<<"del="<<aa<<endl;
			break;
        }
        else
        {
            Eta1_old = Eta1;
        }
        iter++;
    } // end while
  cout << "iter" << iter << endl;
} // end funcs

// Date: 2019-1-7 17:49:53
//' subgroup analysis using admm 
//'
//' @param Y -- num_sample x 1
//' @param X -- num_sample x 1
//' @param Z -- num_dim x num_sample
//' @param Eta1 -- num_dim1 x num_k
//'
//' @return A list
//' 
//' @return A list
//' 
//' @export
// [[Rcpp::export]]
SEXP subG_ADMM_extension(SEXP Yin, SEXP Xin, SEXP Win, SEXP Zin, SEXP Eta1in, SEXP lambdain, SEXP admm_iterin, SEXP tolin){
    try
    {                         // *admm Algorithm
        arma::vec Y = as<arma::vec>(Yin); // *a vector response, dim = num_sample x 1
        arma::vec X = as<arma::vec>(Xin); // *a vector we are interested, dim = num_sample x 1
        arma::mat Z = as<arma::mat>(Zin); // *dim num_sample x q1
        arma::mat W = as<arma::mat>(Win); // covariate matrix, num_sample x c

        int admm_iter = Rcpp::as<int>(admm_iterin);
        double lambda = Rcpp::as<double>(lambdain);
        double tol = Rcpp::as<double>(tolin);
        const int num_sample = Y.n_elem;
        const int num_var = Z.n_cols;

        //================================================
        // compute the Q using Z0 in the manuscript
        // Q -- num_sample x num_sample
        arma::mat Q = W * inv_sympd(W.t() * W) * W.t();
		
		//cout << "Q" << endl;

        //================================================
        // compute Z -- num_sample x num_var
        Z = Z.each_col() % X;
        //arma::mat I = eye<arma::mat>(num_sample, num_sample);
        // *Z_d -- num_sample x q2*num_sample
        //arma::mat Z_d = RowWiseKronProd(I, Z);

		//cout << "Z" << endl;
        //=============================================
        // compute V -- num_sample*(num_sample - 1)/2 x num_sample
        arma::mat I = eye<arma::mat>(num_sample, num_sample);
        arma::mat V = zeros<arma::mat>(num_sample * (num_sample - 1) * 0.5, num_sample);
        size_t t = 0;
        for (size_t i = 0; i < (num_sample - 1); i++)
        {
            for (size_t j = i + 1; j < num_sample; j++)
            {
                V.row(t) = I.row(i) - I.row(j);
                t++;
            } // end for
        }     // end for
        // num_sample x num_sample
        arma::mat VtV = V.t() * V;
		
		//cout << "VtV" << endl;
		
        //arma::mat I2 = eye<arma::mat>(num_var, num_var);
        //arma::mat v_star = KronProd(I2, V);
        //======================================================
        // Compute Eta1, initialize Eta1 -- num_sample x num_var

        //arma::mat Eta1 = zeros<arma::mat>(num_sample, num_var);
        arma::mat Eta1 = as<arma::mat>(Eta1in);
        arma::mat IsQ = eye(size(Q)) - Q;
        InitializeEta1(Z, Y, IsQ, VtV, Eta1);
        //cout<<"Eta1 = " << Eta1 <<endl;
        //========================================
        // Compute alpha
        arma::vec Y_tmp = zeros<arma::vec>(num_sample);
        for (size_t i = 0; i < num_var; i++)
        {
            Y_tmp += Z.col(i) % Eta1.col(i);
        } // end for
        arma::vec alpha = inv(W.t() * W) * W.t() * (Y - Y_tmp);

		cout << "alpha" << endl;
		
        //================================================
        // main loop
        arma::vec rho = ones<arma::vec>(2);
        arma::mat Theta = zeros<arma::mat>(num_var, num_sample * (num_sample - 1) * 0.5);
        arma::mat T = zeros<arma::mat>(num_var, num_sample * (num_sample - 1) * 0.5);
        arma::mat T_old = T;
        //cout<<"here5"<<endl;
        size_t iter = 0;
        double indicator = 0.0;
        double gamma = 2.0;
        while (iter < admm_iter)
        {
            cout << "admm_step: iter = " << iter + 1 << endl;
            //=======================================================
            // *update theta
            // B -- num_var x num_sample*(num_sample-1)/2
            //cout << "admm_step: update theta " << endl;
            arma::mat B = Eta1.t() * V.t() - T / rho(0);
            for (size_t i = 0; i < B.n_cols; i++)
            {
                double sign = 1.0;
                if (norm(B.col(i), 2) > lambda * gamma)
                {
                    Theta.col(i) = B.col(i);
                }
                else
                {
                    indicator = 1 - lambda * (1.0 / rho(0)) / norm(B.col(i), 2);
                    if (indicator > 0)
                    {
                        Theta.col(i) = indicator * Theta.col(i)/(1 - 1.0/gamma/rho(0) );
                    }
                    else
                    {
                        Theta.col(i).zeros();
                    } // end fi
                }     // end fi
            }         // end func

            //cout<<"Theta = "<<Theta.t()<<endl;

            //=====================================
            // *update Eta1
            //cout << "admm_step: update Eta1 " << endl;
            Y_tmp.zeros();
            for (size_t i = 0; i < num_var; i++)
            {
                Y_tmp += Z.col(i) % Eta1.col(i);
            } // end for

            for (size_t i = 0; i < num_var; i++)
            {
                arma::vec Y_iter = Y - ( Y_tmp - Z.col(i) % Eta1.col(i) );
                arma::mat Zdiag = diagmat(Z.col(i));
                arma::vec Tcol = conv_to<arma::vec>::from(T.row(i));
                arma::vec Thetacol = conv_to<arma::vec>::from(Theta.row(i));
                Eta1.col(i) = inv(Zdiag.t() * IsQ * Zdiag + rho(0) * VtV) * (Zdiag * IsQ * Y_iter + V.t() * Tcol + rho(0) * V.t() * Thetacol);
            } // end for
            // beta = inv(Z_d.t() * (eye(size(Q)) - Q) * Z_d + rho(0) * VtV) * (Z_d.t() * (eye(size(Q)) - Q) * Y + v_tide.t() * T + rho(0) * v_tide.t() * Theta);
            //cout<<"Sigma="<<Sigma.t()<<endl;

            //===========================================
            // *update alpha
            //cout << "admm_step: update alpha " << endl;
            Y_tmp.zeros();
            for (size_t i = 0; i < num_var; i++)
            {
                Y_tmp += Z.col(i) % Eta1.col(i);
            } // end for
            alpha = inv(W.t() * W) * W.t() * (Y - Y_tmp);
            //cout<<"alpha="<<alpha.t()<<endl;
            //===========================================
            // *update T
            //cout << "admm_step: update T " << endl;
            T = T_old + rho(0) * (Theta - Eta1.t() * V.t());
            //cout<<"T="<<T.t()<<endl;

            // *stop rule
            // hist_gradT[iter] = norm(gradT, 2);
            cout << "T = " << norm(T - T_old, 2) << endl;
            if(norm(T - T_old, 2) < 1e-3){break;}
            T_old = T;
            iter++;
            //if(norm(gradT, 2) < tol){break;}
        } // *end for admm_iter

        return List::create(Named("Eta1") = Eta1, Named("alpha") = alpha, Named("T") = T, Named("Theta") = Theta, Named("num_iter") = iter - 1);
    }
    catch (std::exception &ex)
    {
        forward_exception_to_r(ex);
    }
    catch (...)
    {
        ::Rf_error("C++ exception (unknown reason)...");
    }
    return R_NilValue;
} // end func


//' @return A list
//' 
//' @export
// [[Rcpp::export]]
SEXP subG_test(SEXP Yin, SEXP Xin)
{
    try
    {                         // *EM Algorith
        arma::vec Y = as<arma::vec>(Yin); // *a vector response, dim = num_sample x 1
        arma::vec X = as<arma::vec>(Xin); // *a vector we are interested, dim = num_sample x 1
        arma::mat K = kron(X, Y);

        return List::create(Named("res_kron") = K);
    }
    catch (std::exception &ex)
    {
        forward_exception_to_r(ex);
    }
    catch (...)
    {
        ::Rf_error("C++ exception (unknown reason)...");
    }
    return R_NilValue;
} // end func
