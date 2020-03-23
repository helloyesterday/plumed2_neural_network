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

#include "NNCV.h"
#include "ActionSet.h"
#include "PlumedMain.h"
#include "tools/DynetTools.h"

namespace PLMD {
	
void NNCV::registerKeywords(Keywords& keys) {
	keys.add("compulsory","MODEL","the model to build neural network");
	keys.add("compulsory","OUTPUT_DIM","1","the dimension of the output of neural network");
    keys.add("optional","PARAMETER_FILE","the file to read the parameter of neural network");
    keys.addFlag("NO_DERIVATIVE",false,"do not calculate the derivative of input w.r.t the neural network. DO NOT USE this flag when the CV is used for the input of an bias potential.");
}

NNCV::NNCV(const ActionOptions&ao):
  Action(ao),
  _is_nn_linked(false),
  _is_bias_linked(false),
  nn_ptr(NULL)
{
	_is_calc_deriv=true;
	bool is_no_deriv;
	parseFlag("NO_DERIVATIVE",is_no_deriv);
	_is_calc_deriv=!is_no_deriv;

	parse("MODEL",nn_label);
	nn_ptr=plumed.getActionSet().selectWithLabel<Neural_Network*>(nn_label);

	if(!nn_ptr)
		plumed_merror("Neural network \""+nn_label+"\" does not exist. NNCV should always be defined AFTER neural network.");
	nn_ptr->linkCV(this);
	_is_nn_linked=true;

	parse("OUTPUT_DIM",output_dim);
	plumed_massert(output_dim>0,"OUTPUT_DIM must be larger than 0!");	
	nn_ptr->set_output_dim(output_dim);
	
	if(keywords.exists("PARAMETER_FILE"))
		parse("PARAMETER_FILE",param_file);
	
	if(is_no_deriv)
		log.printf("  without calculating derivative\n");
	log.printf("  with output dimension: %d\n",int(output_dim));
	if(param_file.size())
		log.printf("  reading parameter of neural network from file: %s\n",param_file.c_str());
}



}

#endif
