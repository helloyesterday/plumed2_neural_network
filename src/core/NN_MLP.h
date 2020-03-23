/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#ifndef __PLUMED_NN_MLP_h
#define __PLUMED_NN_MLP_h

#include "Neural_Network.h"
#include "tools/DynetTools.h"

namespace PLMD {

class NN_MLP :  public Neural_Network
{
private:
	unsigned nhidden;
	std::vector<unsigned> layer_dim;
	std::vector<dytools::Activation> layer_act_funs;
	dytools::MLP mlp;
public:
	static void registerKeywords(Keywords&);
	explicit NN_MLP(const ActionOptions&ao);
	
	void calculate(){}
	void update(){}
	
	void clip(float left,float right,bool clip_last=false) {mlp.clip(left,right,clip_last);}
	void clip_inplace(float left,float right,bool clip_last=false) {mlp.clip_inplace(left,right,clip_last);}
	
	void build_neural_network();
	dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x)
		{return mlp.run(x,cg);}
};


}

#endif
#endif
