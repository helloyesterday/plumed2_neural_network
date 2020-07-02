/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2018-2020 The AIMM code team
   (see the PEOPLE-AIMM file at the root of this folder for a list of names)

   See https://github.com/helloyesterday for more information.

   This file is part of AIMM code module.

   The AIMM code module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The AIMM code module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the AIMM code module.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#ifndef __PLUMED_Trainer_Algorithm_h
#define __PLUMED_Trainer_Algorithm_h

#include "core/Action.h"
#include "core/ActionRegister.h"
#include "Neural_Network.h"
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/training.h>

namespace PLMD {
namespace aimm {

class Neural_Network;

class Trainer_Algorithm :
  public Action
{
protected:
	bool do_clip;
	bool do_clip_last;
	bool use_default;
	
	float learn_rate;
	float clip_left;
	float clip_right;
	float clip_threshold;
	
	std::string algorithm;
	std::string opt_fullname;
	
	std::vector<float> learn_rates;
	std::vector<float> hyper_params;
	std::vector<float> all_params;
	
	Action* bias_ptr;

public:
	static void registerKeywords(Keywords&);
	explicit Trainer_Algorithm(const ActionOptions&ao);
	
	void apply() {};
	void calculate() {};
  
	float learning_rate() const {return learn_rate;}
	std::string algorithm_name() const {return algorithm;}
	
	void nn_clip(Neural_Network* nn) const {if(do_clip) nn->clip_inplace(clip_left,clip_right,do_clip_last);}
	
	dynet::Trainer* new_trainer(dynet::ParameterCollection& pc);
};

}
}

#endif
#endif
