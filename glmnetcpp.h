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
        
    
public:
    // constructor
    glmnetcpp(Eigen::MatrixXd A, Eigen::VectorXd b, double alpha, int num_lambda, int glm_type);
    
    // function to compute the negative log-likelihood (NLL) of exponential GLM from data
    double ExpNegativeLogLikelihood(Eigen::VectorXd x);
    
    // function to compute the gradient of the negative log-likelihood of exponential GLM
    Eigen::VectorXd GradExpNegativeLogLikelihood(Eigen::VectorXd x);
    Eigen::MatrixXd GetA_() const {
        return A_;
    }

    void SetA_(Eigen::MatrixXd A_) {
        this->A_ = A_;
    }

    double GetAlpha_() const {
        return alpha_;
    }

    void SetAlpha_(double alpha_) {
        this->alpha_ = alpha_;
    }

    Eigen::VectorXd GetB_() const {
        return b_;
    }

    void SetB_(Eigen::VectorXd b_) {
        this->b_ = b_;
    }

    int GetGlm_type() const {
        return glm_type_;
    }

    void SetGlm_type(int glm_type) {
        this->glm_type_ = glm_type;
    }

    int GetNum_lambda_() const {
        return num_lambda_;
    }

    void SetNum_lambda_(int num_lambda_) {
        this->num_lambda_ = num_lambda_;
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
    
private:
    // A is the matrix of the independent variables
    Eigen::MatrixXd A_;
    
    // b is the vector of dependent variables
    Eigen::VectorXd b_;
    
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

