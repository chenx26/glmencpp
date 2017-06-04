/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   glmnetcpp.h
 * Author: Xin Chen
 *
 * Created on June 2, 2017, 6:39 PM
 */

#ifndef GLMNETCPP_H
#define GLMNETCPP_H
#include <Eigen/Dense>

class glmnetcpp{
private:
    // A is the matrix of the independent variables
    Eigen::MatrixXd A_;
    
    // b is the vector of dependent variables
    Eigen::VectorXd b_;
    
    // alpha is the weight between L1 and L2 regularization, between 0 and 1.
    double alpha_;
    
    // num_lambda is the number of lambdas for the search grid
    int num_lambda_;
    
    
public:
    // constructor
    glmnetcpp(Eigen::MatrixXd A, Eigen::VectorXd b, double alpha, int num_lambda);
    
    // function to compute the negative log-likelihood (NLL) of exponential GLM from data
    double ExpNegativeLogLikelihood();
    
    // function to compute the gradient of the negative log-likelihood of exponential GLM
    Eigen::VectorXd GradExpNegativeLogLikelihood();
    
        // function to compute the negative log-likelihood (NLL) of Gamma GLM from data
    double GammaNegativeLogLikelihood();
    
    // function to compute the gradient of the negative log-likelihood of Gamma GLM
    Eigen::VectorXd GradGammaNegativeLogLikelihood();
    
    // function for the soft-thresholding operator, this is multi-dimensional
    Eigen::VectorXd SoftThresholding();
    
    // function for the smooth part of the objective function
    double SmoothObjFun();
    
    // function for the gradient of the smooth part of the objective function
    Eigen::VectorXd GradSmoothObjFun();
    
    // function for performing Proximal Gradient Descent (PGD)
    Eigen::VectorXd ProxGradDescent();

};




#endif /* GLMNETCPP_H */

