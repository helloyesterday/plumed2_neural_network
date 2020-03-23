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

#ifndef __PLUMED_bias_NN_Bias_h
#define __PLUMED_bias_NN_Bias_h

#include <random>
#include "Bias.h"
#include "core/NNCV.h"

namespace PLMD{
namespace bias{

class NN_Bias: public Bias
{
protected:
	double temp;
	double beta;
	double kB;
	double kBT;
	double scale_factor;
	double energy_scale;
	
	std::string nncv_label;
	NNCV* nncv_ptr;
public:
	explicit NN_Bias(const ActionOptions&);
	~NN_Bias();
	void calculate();
	virtual void update(){};
	static void registerKeywords(Keywords& keys);
	
	NNCV* get_nncv_ptr() {return nncv_ptr;}
	
	Neural_Network* get_nn_ptr() {return nncv_ptr->get_nn_ptr();}
	dynet::ParameterCollection& get_nn_model() {return nncv_ptr->get_nn_model();}
	
	unsigned get_cvs_number() {return nncv_ptr->get_input_number();}
	unsigned get_input_dim() {return nncv_ptr->get_input_dim();}
	unsigned get_output_dim() {return nncv_ptr->get_output_dim();}
	
	std::string get_nncv_label() const {return nncv_label;}
	std::vector<float> get_cv_input() {return nncv_ptr->get_input_values();}
	std::vector<float> get_nn_input() {return nncv_ptr->get_input_layer();}
};

}
}

#endif
#endif
