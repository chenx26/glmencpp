
// #include <iostream>
// #include <Eigen/Dense>
// #include "glmnetcpp.h"
#include "GlmNetCvCpp.h"

int main()
{
    int num_obs = 4;
    int num_params = 3;
    double lambda = 1;
    double alpha = 0.5;
    Eigen::MatrixXd predictor_matrix = Eigen::MatrixXd::Random(num_obs, num_params);
    Eigen::VectorXd response_vector = Eigen::VectorXd::Random(num_obs);
    Eigen::VectorXd init_sol = Eigen::VectorXd::Random(num_params);
    GlmNetCpp test_glm(predictor_matrix, response_vector, alpha = alpha);
    
//    // print original inputs
    std::cout << predictor_matrix << std::endl;
    std::cout << response_vector << std::endl;
//    
//    // print private members of the glm object
    std::cout << test_glm.get_alpha() << std::endl;
    std::cout << test_glm.get_glm_type() << std::endl;
    std::cout << test_glm.get_predictor_matrix() << std::endl;
    std::cout << test_glm.get_response_vector() << std::endl;
//    
//    // print the outputs of helper functions
    std::cout << test_glm.ExpNegativeLogLikelihood(init_sol) << std::endl;
    std::cout << test_glm.GradExpNegativeLogLikelihood(init_sol) << std::endl;
    std::cout << test_glm.SmoothObjFun(init_sol, lambda) << std::endl;
    std::cout << test_glm.GradSmoothObjFun(init_sol, lambda) << std::endl;
    std::cout << Eigen::VectorXd::LinSpaced(5,0,1).maxCoeff() << std::endl;
    std::cout << test_glm.ProxGradDescent(lambda) << std::endl;
//    std::cout << (Eigen::VectorXd::LinSpaced(5, log(0.001), log(test_glm.ComputeLambdaMax()))).array().exp().transpose() << std::endl;
//    std::cout << test_glm.GenerateLambdaGrid() << std::endl;
    
//      std::array<int,5> foo {1,2,3,4,5};
//    Eigen::VectorXd foo1 = Eigen::VectorXd::LinSpaced(num_obs - 1, 0, num_obs - 2);
//    std::vector<int> foo(foo1.data(), foo1.data() + foo1.size());
//    
//      std::cout << "original elements:";
//  for (int& x: foo) std::cout << ' ' << x;
//  std::cout << '\n';
//    
//  // obtain a time-based seed:
//  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//
//  shuffle (foo.begin(), foo.end(), std::default_random_engine(seed));
//
//  std::cout << "shuffled elements:";
//  for (int& x: foo) std::cout << ' ' << x;
//  std::cout << '\n';
//  
//  std::cout << (int)(3.5/2) << std::endl;
//  
//  // Build the randmom matrix
//  
//  Eigen::MatrixXd par1;
//  Eigen::VectorXd par2;
//  std::tie(par1, par2) = std::make_tuple(predictor_matrix, response_vector);
//  std::cout <<  par1 << std::endl;
  



////////////////////////////////
////////////////////////////////    
//  std::cout << ((m.array() - 1).sign()).matrix() << std::endl;
  
//  Eigen::MatrixXd m(2,2);
//  Eigen::MatrixXd n(2,2);
//  m(0,0) = 3;
//  m(1,0) = 2.5;
//  m(0,1) = -1;
//  m(1,1) = m(1,0) + m(0,1);
//  
//  n(0,0) = 4;
//  n(1,0) = 4;
//  n(0,1) = 4;
//  n(1,1) = 4;
//  m = m.array().exp();
}