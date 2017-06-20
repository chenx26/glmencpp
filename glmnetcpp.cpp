/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "glmnetcpp.h"

glmnetcpp::glmnetcpp(Eigen::MatrixXd A, Eigen::VectorXd b, double alpha = 1,
        int num_lambda = 100, int glm_type = 1) {
    this->A_ = A;
    this->b_ = b;
    this->alpha_ = alpha;
    this->num_lambda_ = num_lambda;
    this->glm_type_ = glm_type;
}

double glmnetcpp::ExpNegativeLogLikelihood(Eigen::VectorXd x) {
    // compute the linear component
    Eigen::VectorXd rs = A_ * x;

    // return the negative log-likelihood
    return rs.sum() + b_.transpose() * (-rs).array().exp().matrix();
}

// function to compute the gradient of the negative log-likelihood of exponential GLM

Eigen::VectorXd glmnetcpp::GradExpNegativeLogLikelihood(Eigen::VectorXd x) {
    // number of variables
    int p = A_.cols();

    // number of observations
    int n = A_.rows();

    // create vector of n 1s
    Eigen::VectorXd my_ones = Eigen::VectorXd::Ones(n);

    // the gradient of the rs.sum() term in the NLL
    Eigen::VectorXd grad = A_.transpose() * my_ones;

    // compute the linear component
    Eigen::VectorXd rs = A_ * x;

    // the gradient of the b_.transpose() * (-rs).array().exp().matrix() term
    grad += (-A_.transpose()) * ((-rs).array().exp().matrix()).cwiseProduct(b);

    return grad;

}

// function to compute the negative log-likelihood (NLL) of Gamma GLM from data
double GammaNegativeLogLikelihood(Eigen::VectorXd x);

// function to compute the gradient of the negative log-likelihood of Gamma GLM
Eigen::VectorXd GradGammaNegativeLogLikelihood(Eigen::VectorXd x);

// function for the soft-thresholding operator, this is multi-dimensional

Eigen::VectorXd glmnetcpp::SoftThresholding(Eigen::VectorXd x, double threshold) {
    return ((abs(x.array()) - threshold).max(0) * x.array().sign()).matrix();
}

// function for the smooth part of the objective function

double glmnetcpp::SmoothObjFun(Eigen::VectorXd x, double lambda) {
    return glmnetcpp::ExpNegativeLogLikelihood(x) +
            lambda * (1 - alpha_) / 2 * x.squaredNorm();
}

// function for the gradient of the smooth part of the objective function

Eigen::VectorXd GradSmoothObjFun(Eigen::VectorXd x, double lambda) {
    return glmnetcpp::GradExpNegativeLogLikelihood(x) +
            (lambda * (1 - alpha_) * x.array()).matrix();
}

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
Eigen::MatrixXd GlmNetCpp::get_predictor_matrix(){
    return predictor_matrix_;
}

Eigen::VectorXd GlmNetCpp::get_response_vector(){
    return response_vector_;
}

double GlmNetCpp::get_alpha(){
    return alpha_;
}

int GlmNetCpp::get_num_lambda(){
    return num_lambda_;
}

int GlmNetCpp::get_glm_type(){
    return glm_type_;
}

// set functions
void GlmNetCpp::set_predictor_matrix(Eigen::MatrixXd M){
    predictor_matrix_ = M;
}

void set_response_vector(Eigen::VectorXd V){
    response_vector_ = V;
}

void set_alpha(double x){
    alpha_ = x;
}

void set_num_lambda(int x){
    num_lambda_ = x;
}

void set_glm_type(int x){
    glm_type_ = x;
}


