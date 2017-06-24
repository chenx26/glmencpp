/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "glmnetcpp.h"

GlmNetCpp::GlmNetCpp(const Eigen::MatrixXd& predictor_matrix, 
        const Eigen::VectorXd& response_vector, double alpha,
        int num_lambda, int glm_type,
        int max_iter,
        double abs_tol,
        double rel_tol): 
        predictor_matrix_(predictor_matrix),
        response_vector_(response_vector),
        alpha_(alpha),
        num_lambda_(num_lambda),
        glm_type_(glm_type),
        max_iter_(max_iter),
        abs_tol_(abs_tol),
        rel_tol_(rel_tol){
//    predictor_matrix_ = A;
//    response_vector_ = b;
//    alpha_ = alpha;
//    num_lambda_ = num_lambda;
//    glm_type_ = glm_type;
}

double GlmNetCpp::ExpNegativeLogLikelihood(const Eigen::VectorXd& x) {
    // compute the linear component
    Eigen::VectorXd rs = predictor_matrix_ * x;

    // return the negative log-likelihood
    return rs.sum() + response_vector_.transpose() * (-rs).array().exp().matrix();
}

// function to compute the gradient of the negative log-likelihood of exponential GLM

Eigen::VectorXd GlmNetCpp::GradExpNegativeLogLikelihood(const Eigen::VectorXd& x) {
    // number of variables
    int p = predictor_matrix_.cols();

    // number of observations
    int n = predictor_matrix_.rows();

    // create vector of n 1s
    Eigen::VectorXd my_ones = Eigen::VectorXd::Ones(n);

    // the gradient of the rs.sum() term in the NLL
    Eigen::VectorXd grad = predictor_matrix_.transpose() * my_ones;

    // compute the linear component
    Eigen::VectorXd rs = predictor_matrix_ * x;

    // the gradient of the response_vector__.transpose() * (-rs).array().exp().matrix() term
    grad += (-predictor_matrix_.transpose()) * ((-rs).array().exp().matrix()).cwiseProduct(response_vector_);

    return grad;

}

// function to compute the negative log-likelihood (NLL) of Gamma GLM from data
double GlmNetCpp::GammaNegativeLogLikelihood(const Eigen::VectorXd& x){
    return 0;
}

// function to compute the gradient of the negative log-likelihood of Gamma GLM
Eigen::VectorXd GradGammaNegativeLogLikelihood(const Eigen::VectorXd& x){
    return Eigen::VectorXd::Zero(3);
}

// function for the soft-thresholding operator, this is multi-dimensional

Eigen::VectorXd GlmNetCpp::SoftThresholding(const Eigen::VectorXd& x, double threshold) {
    return ((abs(x.array()) - threshold).max(0) * x.array().sign()).matrix();
}

// function for the smooth part of the objective function

double GlmNetCpp::SmoothObjFun(const Eigen::VectorXd& x, double lambda) {
    return GlmNetCpp::ExpNegativeLogLikelihood(x) +
            lambda * (1 - alpha_) / 2 * x.squaredNorm();
}

// function for the gradient of the smooth part of the objective function

Eigen::VectorXd GlmNetCpp::GradSmoothObjFun(const Eigen::VectorXd& x, 
        double lambda) {
    return GlmNetCpp::GradExpNegativeLogLikelihood(x) +
            (lambda * (1 - alpha_) * x.array()).matrix();
}

// function for performing Proximal Gradient Descent (PGD)
// This is in fact based on the Fast Proximal gradient at
// https://web.stanford.edu/~boyd/papers/prox_algs/lasso.html#9

Eigen::VectorXd GlmNetCpp::ProxGradDescent(double lambda){
    
    double t = 1;
    double beta = 0.5;
    
    int num_params = predictor_matrix_.cols();
    
    Eigen::VectorXd x = Eigen::VectorXd::Zero(num_params);
    Eigen::VectorXd xprev = x;
    Eigen::VectorXd z;
    
    for(int k = 0; k < max_iter_; k++){
        Eigen::VectorXd y = x + (k/(k+3)) * (x - xprev);
//        std::cout << "y = " << y << std::endl;
        while(1){
            Eigen::VectorXd grad_y = GradSmoothObjFun(y, lambda);
            
//            std::cout << "grad_y =" << grad_y << std::endl;
            
            z = SoftThresholding(y - t * grad_y,
                    t * lambda);
//            std::cout << "z = " << z << std::endl;
            
            double lhs = SmoothObjFun(z, lambda);
//            std::cout << "lhs =" << lhs << std::endl;
            
            double rhs1 = SmoothObjFun(y, lambda);
//            std::cout << "rhs1 =" << rhs1 << std::endl;
            
            double rhs2 = grad_y.transpose() * (z - y);
//            std::cout << "rhs2 =" << rhs2 << std::endl;
            
            double rhs3 = (1/(2*t)) * (z - y).squaredNorm();
//            std::cout << "rhs3 =" << rhs3 << std::endl;
//            
//            std::cout << (lhs <= rhs1 +rhs2 + rhs3) << std::endl;
            
            if (lhs <=
                    rhs1 + 
                    rhs2 +
                    rhs3){
                std::cout << "breaking out" << std::endl;
                break;
            }
            
//            std::cout << "t = " << t << std::endl;
//            std::cout << "beta = " << beta << std::endl;
            t = beta * t;
            
        }
        xprev = x;
        x = z;
    }
    return x;
}

// function for fitting GLM model given fixed lambda
Eigen::VectorXd GlmNetCpp::FitGlmFixed(){
    return Eigen::VectorXd::Zero(3);
}

// function for generating a grid of candidate lambdas
Eigen::VectorXd GlmNetCpp::GenerateLambdaGrid(){
    return Eigen::VectorXd::Zero(3);
}

// function for automatically choosing the optimal lambda 
// and the corresponding weights using cross validation
Eigen::VectorXd GlmNetCpp::FitGlmCv(){
    return Eigen::VectorXd::Zero(3);
}

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
//void GlmNetCpp::set_predictor_matrix(Eigen::MatrixXd M){
//    predictor_matrix_ = M;
//}
//
//void GlmNetCpp::set_response_vector(Eigen::VectorXd V){
//    response_vector_ = V;
//}

void GlmNetCpp::set_alpha(double x){
    alpha_ = x;
}

void GlmNetCpp::set_num_lambda(int x){
    num_lambda_ = x;
}

void GlmNetCpp::set_glm_type(int x){
    glm_type_ = x;
}


