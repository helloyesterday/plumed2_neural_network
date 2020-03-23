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

#ifndef __DYNET_NN_MLP_H
#define __DYNET_NN_MLP_H

#include "Function.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace function {
	
using namespace dytools;

//+PLUMEDOC FUNCTION NEURAL_NETWORK
/*

*/
//+ENDPLUMEDOC


class NN_MLP :	public Function
{
private:
	unsigned narg;
	unsigned nlayer;
	unsigned random_seed;
	
	bool init_dynet;
	bool firsttime;
	bool _is_linked;

	std::string param_read_file;
	std::string param_output_file;

	std::vector<unsigned> layer_dims;
	std::vector<std::string> str_afs;
	std::vector<Activation> act_funs;
	std::vector<std::vector<dynet::Parameter>> params;

	dynet::ParameterCollection pc;
	MLP nn;

public:
	explicit NN_MLP(const ActionOptions&);
	static void registerKeywords(Keywords& keys);
	void calculate();
	void prepare();
	
	bool is_linked() const {return _is_linked;}
	unsigned get_random_seed() const {return random_seed;}

	//~ void set_parameters();

	dynet::Expression energy(dynet::ComputationGraph& cg,const dynet::Expression& inputs){
		return nn.run(inputs,cg);
	}
};


}
}

#endif
#endif
