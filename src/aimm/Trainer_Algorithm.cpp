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

#include "Trainer_Algorithm.h"
#include <dynet/io.h>
#include "DynetTools.h"

namespace PLMD {
namespace aimm {
	
//+PLUMEDOC AIMM OPT_ALGORITHM
/*

*/
//+ENDPLUMEDOC
	
PLUMED_REGISTER_ACTION(Trainer_Algorithm,"OPT_ALGORITHM")

void Trainer_Algorithm::registerKeywords(Keywords& keys) {
	Action::registerKeywords(keys);
	keys.add("compulsory","ALGORITHM","the algorithm to train the neural networks");
	keys.add("optional","LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","HYPERPARAMETERS","other hyperparameters for training the neural network");
	keys.add("optional","CLIP_THRESHOLD","the clip threshold for training the neural network");
	keys.add("optional","CLIP_RANGE","the left value of the range to clip");
	keys.addFlag("CLIP_LAST_LAYER",false,"clip the last layer of the neural network");
}

Trainer_Algorithm::Trainer_Algorithm(const ActionOptions&ao):
  Action(ao),
  bias_ptr(NULL)
{
	parse("ALGORITHM",algorithm);
	
	use_default=true;
	if(keywords.exists("LEARN_RATE"))
	{
		use_default=false;
		parseVector("LEARN_RATE",learn_rates);
		learn_rate=learn_rates[0];
	}
	
	if(keywords.exists("HYPERPARAMETERS"))
	{
		plumed_massert(!use_default,"HYPERPARAMETERS must be used with LEARN_RATE.");
		parseVector("LEARN_RATE",hyper_params);
	}

	clip_threshold=-1;
	if(keywords.exists("CLIP_THRESHOLD"))
	{
		parse("CLIP_THRESHOLD",clip_threshold);
		plumed_massert(clip_threshold>0,"CLIP_THRESHOLD must be larger than ZERO.");
	}
	
	parseFlag("CLIP_LAST_LAYER",do_clip_last);
	
	do_clip=false;
	if(keywords.exists("CLIP_RANGE"))
	{
		do_clip=true;
		std::vector<float> clip_range;
		parseVector("CLIP_RANGE",clip_range);
		plumed_massert(clip_range.size()==2,"CLIP_RANGE must have two values: left and right");
		clip_left=clip_range[0];
		clip_right=clip_range[1];
		plumed_massert(clip_right>clip_left,"CLIP_LEFT should be less than CLIP_RIGHT");
	}
	
	checkRead();
	
	dynet::ParameterCollection pc;
	dynet::Trainer* trainer=new_trainer(pc);
	learn_rate=trainer->learning_rate;
	if(clip_threshold>0)
		trainer->clip_threshold = clip_threshold;
	clip_threshold=trainer->clip_threshold;
	delete trainer;
	
	log.printf("  with optimation algorithm: %s\n",opt_fullname.c_str());
	log.printf("  with learning rate: %f\n",learn_rate);
	if(use_default)
		log.printf("  with default hyperparameters\n");
	else
	{
		log.printf("  with manual hyperparameters:");
		for(unsigned i=0;i!=hyper_params.size();++i)
			log.printf(" %f",hyper_params[i]);
		log.printf("\n");
	}
	log.printf("  with clip threshold: %f\n",clip_threshold);
	if(do_clip)
	{
		log.printf("  with cliping the parameter of neural network range from %f to %f\n",clip_left,clip_right);
		if(do_clip_last)
			log.printf("  with cliping the last layer of neural network");
		else
			log.printf("  without cliping the last layer of neural network");
	}
}

dynet::Trainer* Trainer_Algorithm::new_trainer(dynet::ParameterCollection& pc)
{
	if(use_default)
	{
		return new_traniner(algorithm,pc,opt_fullname);
	}
	else
	{
		if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
		{
			plumed_massert(learn_rates.size()==2,"The CyclicalSGD algorithm need two learning rates");
		}
		else
		{
			plumed_massert(learn_rates.size()==1,"The "+algorithm+" algorithm need only one learning rates");
		}
		
		std::vector<float> all_params=learn_rates;
		if(hyper_params.size()>0)
			all_params.insert(all_params.end(),hyper_params.begin(),hyper_params.end());
		
		return new_traniner(algorithm,pc,all_params,opt_fullname);
	}
}


}
}

#endif
