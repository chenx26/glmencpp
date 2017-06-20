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

class GlmNetCpp{
        
    
public:
    // constructor
    GlmNetCpp(Eigen::MatrixXd predictor_matrix, Eigen::VectorXd b, double alpha, int num_lambda, int glm_type);
    
    // function to compute the negative log-likelihood (NLL) of exponential GLM from data
    double exp_negative_log_likelihood(Eigen::VectorXd x);
    
    // function to compute the gradient of the negative log-likelihood of exponential GLM
    Eigen::VectorXd grad_exp_negative_log_likelihood(Eigen::VectorXd x);
    Eigen::MatrixXd get_predictor_matrix() const {
        return predictor_matrix_;
    }

        // function to compute the negative log-likelihood (NLL) of Gamma GLM from data
    double GammaNegativeLogLikelihood(Eigen::VectorXd x);
    
    // function to compute the gradient of the negative log-likelihood of Gamma GLM
    Eigen::VectorXd GradGammaNegativeLogLikelihood(Eigen::VectorXd x);
    
    // function for the soft-thresholding operator, this is multi-dimensional
    Eigen::VectorXd SoftThresholding(Eigen::VectorXd x);
    
    // function for the smooth part of the objective function
    double SmoothObjFun(Eigen::VectorXd x);
    
    // function for the gradient of the smooth part of the objective function
    Eigen::VectorXd GradSmoothObjFun(Eigen::VectorXd x);
    
    // function for performing Proximal Gradient Descent (PGD)
    Eigen::VectorXd ProxGradDescent();
    
    // function for fitting GLM model given fixed lambda
    Eigen::VectorXd FitGlmFixed();
    
    // function for generating a grid of candidate lambdas
    Eigen::VectorXd GenerateLambdaGrid();
    
    // function for automatically choosing the optimal lambda 
    // and the corresponding weights using cross validation
    Eigen::VectorXd FitGlmCv();
    
    // get functions
    Eigen::MatrixXd get_predictor_matrix();
    
    Eigen::VectorXd get_response_vector();
    
    double get_alpha();
    
    int get_num_lambda();
    
    int get_glm_type();
    
    // set functions
    void set_predictor_matrix(Eigen::MatrixXd M);
    
    void set_response_vector(Eigen::VectorXd V);
    
    void set_alpha(double x);
    
    void set_num_lambda(int x);
    
    void set_glm_type(int x);
    
private:
    // predictor_matrix_ is the matrix of the independent variables
    Eigen::MatrixXd predictor_matrix_;
    
    // b is the vector of dependent variables
    Eigen::VectorXd response_vector_;
    
    // alpha is the weight between L1 and L2 regularization, between 0 and 1.
    double alpha_;
    
    // num_lambda is the number of lambdas for the search grid
    int num_lambda_;
    
    // type of GLM
    // 1: Exponential
    // 2: Gamma
    int glm_type_;

};




#endif /* GLMNETCPP_H */

