/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	 Copyright (c) 2011-2017 The plumed team
	 (see the PEOPLE file at the root of the distribution for a list of names)

	 See http://www.plumed.org for more information.

	 This file is part of plumed, version 2.

	 plumed is free software: you can redistribute it and/or modify
	 it under the terms of the GNU Lesser General Public License as published by
	 the Free Software Foundation, either version 3 of the License, or
	 (at your option) any later version.

	 plumed is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
	 GNU Lesser General Public License for more details.

	 You should have received a copy of the GNU Lesser General Public License
	 along with plumed.	If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#include "NN_MLP.h"
#include "ActionRegister.h"
#include "tools/Communicator.h"

#include <cmath>

namespace PLMD {
namespace function {
	
using namespace dytools;

//+PLUMEDOC FUNCTION NEURAL_NETWORK
/*

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(NN_MLP,"NN_MLP")

void NN_MLP::registerKeywords(Keywords& keys) {
	Function::registerKeywords(keys);
	useCustomisableComponents(keys);
	keys.use("ARG");
	keys.add("compulsory","LAYERS_NUMBER","3","the hidden layers of the multilayer preceptron");
	keys.add("compulsory","LAYER_DIMENSIONS","64","the dimension of each hidden layers");
	keys.add("compulsory","ACTIVE_FUNCTIONS","SWISH","the activation function for the neural network");
	keys.add("compulsory","PARAMETERS_OUTPUT","GNN_parameters.data","the file to output the parameters of the neural network");
	keys.add("optional","PARAM_READ_FILE","the file to output the parameters of the neural network");
	keys.addFlag("DYNET_INITIALIZATION",false,"initialize the DyNet. Only do on time when you use the function of dynet");
}

NN_MLP::NN_MLP(const ActionOptions&ao):
Action(ao),
Function(ao),
narg(getNumberOfArguments()),
firsttime(true),
_is_linked(false)
{
	parse("LAYERS_NUMBER",nlayer);
	plumed_massert(nlayer>0,"LAYERS_NUMBER must be larger than 0!");

	parseVector("LAYER_DIMENSIONS",layer_dims);
	if(layer_dims.size()!=nlayer)
	{
		if(layer_dims.size()==1)
		{
			unsigned dim=layer_dims[0];
			layer_dims.assign(nlayer,dim);
		}
		else
			plumed_merror("LAYER_DIMENSIONS mismatch!");
	}

	parseVector("ACTIVE_FUNCTIONS",str_afs);
	if(str_afs.size()!=nlayer)
	{
		if(str_afs.size()==1)
		{
			std::string af=str_afs[0];
			str_afs.assign(nlayer,af);
		}
		else
			plumed_merror("ACTIVE_FUNCTIONS mismatch!");
	}

	act_funs.resize(nlayer);
	for(unsigned i=0;i!=nlayer;++i)
		act_funs[i]=activation_function(str_afs[i]);

	parse("PARAMETERS_OUTPUT",param_output_file);
	parse("PARAM_READ_FILE",param_read_file);
	
	parseFlag("DYNET_INITIALIZATION",init_dynet);

	addValueWithDerivatives();
	setNotPeriodic();
	checkRead();
	
	if(init_dynet)
	{
		if(comm.Get_rank()==0)
		{
			std::random_device rd;
			random_seed=rd();
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);

		dynet_initialization(random_seed);
	}

	log.printf("  with number of input argument: %d\n",int(narg));
	log.printf("  with number of hidden layers: %d\n",int(nlayer));
	for(unsigned i=0;i!=nlayer;++i)
		log.printf("    Hidden layer %d with dimension %d and activation funciton %s\n",int(i),int(layer_dims[i]),str_afs[i].c_str());
	log.printf("  with parameter output file: %s\n",param_output_file.c_str());
	if(param_read_file.size()>0)
		log.printf("  with parameter read file: %s\n",param_read_file.c_str());
}

void NN_MLP::prepare()
{
	if(firsttime)
	{
		unsigned ldim=narg;
		for(unsigned i=0;i!=nlayer;++i)
		{
			nnv.append(pc,Layer(ldim,layer_dims[i],act_funs[i],0));
			ldim=layer_dims[i];
		}
		nnv.append(pc,Layer(ldim,1,Activation::LINEAR,0));
		
		if(param_read_file.size()>0)
		{
			dynet::TextFileLoader loader(param_read_file);
			loader.populate(pc);
		}
		dynet::TextFileSaver saver(param_output_file);
		saver.save(pc);
		
		firsttime=false;
	}
}

void NN_MLP::calculate()
{
	std::vector<float> args(narg);
	for(unsigned i=0;i!=narg;++i)
		args[i]=getArgument(i);

	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&args);

	dynet::Expression output=nn.run(inputs,cg);
	cg.forward(output);
	cg.backward(output,true);

	std::vector<float> outvec=dynet::as_vector(output.value());
	double value=outvec[0];
	setValue(value);

	std::vector<float> deriv=dynet::as_vector(inputs.gradient());
	for(unsigned i=0;i!=narg;++i)
		setDerivative(i,deriv[i]);
}


}
}


#endif
