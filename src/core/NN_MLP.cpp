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

#include "NN_MLP.h"
#include "ActionRegister.h"

namespace PLMD {
	
PLUMED_REGISTER_ACTION(NN_MLP,"MLP")

void NN_MLP::registerKeywords(Keywords& keys) {
	Neural_Network::registerKeywords(keys);
	keys.add("compulsory","LAYER_NUMBER","3","the hidden layers of the multilayer preceptron of the neural networks");
	keys.add("compulsory","LAYER_DIMENSION","32","the dimension of each hidden layers of the neural networks");
	keys.add("compulsory","ACTIVE_FUNCTION","SWISH","the activation function for the neural networks");
}


NN_MLP::NN_MLP(const ActionOptions&ao):
	Neural_Network(ao)
{
	parse("LAYER_NUMBER",nhidden);
	plumed_massert(nhidden>0,"LAYER_NUMBER must be larger than 0!");

	parseVector("LAYER_DIMENSION",layer_dim);
	if(layer_dim.size()!=nhidden)
	{
		if(layer_dim.size()==1)
		{
			unsigned dim=layer_dim[0];
			layer_dim.assign(nhidden,dim);
		}
		else
			plumed_merror("the size of LAYER_DIMENSION mismatch!");
	}
	for(unsigned i=0;i!=nhidden;++i)
		plumed_massert(layer_dim[i]>0,"LAYER_DIMENSION must be larger than 0!");
	
	std::vector<std::string> str_act_funs;
	parseVector("ACTIVE_FUNCTION",str_act_funs);
	if(str_act_funs.size()!=nhidden)
	{
		if(str_act_funs.size()==1) 
		{
			std::string af=str_act_funs[0];
			str_act_funs.assign(nhidden,af);
		}
		else
			plumed_merror("the size of ACTIVE_FUNCTION mismatch!");
	}
	
	layer_act_funs.resize(nhidden);
	for(unsigned i=0;i!=nhidden;++i)
		layer_act_funs[i]=dytools::activation_function(str_act_funs[i]);
	
	checkRead();
	
	log.printf("  using multilayer perceptron (MLP) with %d hidden layers:\n",int(nhidden));
	for(unsigned i=0;i!=nhidden;++i)
		log.printf("    Hidden layer %d with dimension %d and activation funciton %s\n",int(i+1),int(layer_dim[i]),str_act_funs[i].c_str());
}

void NN_MLP::build_neural_network()
{
	plumed_massert(input_dim>0,"the input dimension must be setup before building neural network");
	plumed_massert(output_dim>0,"the output dimension must be setup before building neural network");
	unsigned ldim=input_dim;
	for(unsigned i=0;i!=nhidden;++i)
	{
		mlp.append(pc,dytools::Layer(ldim,layer_dim[i],layer_act_funs[i],0));
		ldim=layer_dim[i];
	}
	mlp.append(pc,dytools::Layer(ldim,get_output_dim(),dytools::Activation::LINEAR,0));
}



}

#endif
