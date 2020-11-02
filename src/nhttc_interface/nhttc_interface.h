/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

MODIFIED BY STEFAN LAYANTO on behalf of the MuSHR Project of the Personal Robotics Lab at the University of Washington
*/

#pragma once
#include <sgd/ttc_sgd_problem.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <functional>

#include <time.h>

#include <Eigen/Core>

// From UTILS
std::vector<std::string> GetParts(std::string s, char delim);
std::vector<std::vector<std::string>> LoadFileByToken(std::string file_name, int n_skip, char delim);

void SetBoundsV(TTCParams &params);
void SetBoundsA(TTCParams &params);
void SetBoundsDD(TTCParams &params);
void SetBoundsADD(TTCParams &params);
void SetBoundsCAR(TTCParams &params);
void SetBoundsACAR(TTCParams &params);
void SetBoundsMUSHR(TTCParams &params);

int GetVector(const std::vector<std::string>& parts, int offset, int v_len, Eigen::VectorXf& v);

// END UTILS


enum class AType { V, A, DD, ADD, CAR, ACAR, MUSHR };

class Agent {
public:
  TTCSGDProblem* prob;
  SGDOptParams opt_params;
  int u_dim, x_dim;
  bool reactive, controlled, done = false;
  AType a_type;
  std::string type_name;
  Eigen::Vector2f goal;

  Agent(SGDOptParams opt_params_in);
  Agent(std::vector<std::string> parts, SGDOptParams opt_params_in);

  // Added for easier interfacing

  void SetPlanTime(float agent_plan_time_ms);

  // Pass in own index if agent is itself one of the obstacles passed in, otherwise pass in -1
  void SetObstacles(std::vector<TTCObstacle*> obsts, size_t own_index);

  // Update Goal by passing in a new vector. If unchanged, pass in NULL. Calling the function will update goal and pass it into SGD
  void UpdateGoal(Eigen::Vector2f new_goal);

  // 
  void SetEgo(Eigen::VectorXf new_x_o);

  // Controls can be extracted with agent->params.ucurr at any time. This function runs the update and also returns the controls.
  Eigen::VectorXf UpdateControls();

  // Original From nhttc_utils
  float GetBestGoalCost(float dt, const Eigen::Vector2f& pos);
  float GetBestCost();
  bool AtGoal();
  void SetStop();
  void PrepareSGDParams();

private:
  int example;

};


std::vector<TTCObstacle*> BuildObstacleList(std::vector<Agent> agents);

void SetAgentObstacleList(Agent& a, size_t a_idx, std::vector<TTCObstacle*> obsts);