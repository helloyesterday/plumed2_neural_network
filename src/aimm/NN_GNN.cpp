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

#include "NN_GNN.h"
#include "core/ActionRegister.h"

namespace PLMD {
namespace aimm {
	
void NN_GNN::registerKeywords(Keywords& keys) {
	Neural_Network::registerKeywords(keys);
	keys.add("compulsory","ACTIVE_FUNCTION","SWISH","the activation function for the neural networks");
	keys.add("compulsory","CUTOFF","3.0","the demension of the vector at the hidden layer of graph neural network");
	keys.add("compulsory","RBF_NUMBER","128","the number of radical basis functions (RBF) to expand the distance bewteen each atoms");
	keys.add("compulsory","VECTOR_DIMENSION","64","the demension of the vector at the hidden layer of graph neural network");
	keys.add("optional","TYPES_NUMBER","the total number of atom types.");
}

NN_GNN::NN_GNN(const ActionOptions&ao):
	Neural_Network(ao),
	natoms(0),
	ntypes(0),
	is_self_dis(false)
{
	parse("ACTIVE_FUNCTION",str_af);
	af=activation_function(str_af);
	
	parse("CUTOFF",cutoff);
	parse("RBF_NUMBER",rbf_num);
	parse("VECTOR_DIMENSION",vec_dim);
	
	parse("TYPES_NUMBER",ntypes);
	
	log.printf("  using activation function: %s\n",str_af.c_str());
	log.printf("  with number of atoms types: %d\n",int(ntypes));
	log.printf("  with cutoff distance: %f\n",cutoff);
	log.printf("  with vector dimension: %d\n",int(vec_dim));
	log.printf("  with number of radical basis functions: %d\n",int(rbf_num));
}


}
}

#endif
