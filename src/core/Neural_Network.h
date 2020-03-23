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

#ifndef __PLUMED_Neural_Network_h
#define __PLUMED_Neural_Network_h

#include "Action.h"
#include <dynet/expr.h>
#include <dynet/model.h>

//~ #define PLUMED_NEURAL_NETWORK_INIT(ao) Action(ao),Neural_Network(ao)

namespace PLMD {

class Neural_Network :
  public Action
{
protected:
	bool _is_cv_linked;
	
	unsigned input_dim;
	unsigned output_dim;
	
	std::string linked_label;
	dynet::ParameterCollection pc;
	
	Action* cv_ptr;
public:
	static void registerKeywords(Keywords&);
	explicit Neural_Network(const ActionOptions&ao);
  
	void apply() {};
	void calculate() {};
	
	bool is_cv_linked() const {return _is_cv_linked;}
	
	void linkCV(Action* _cv_ptr) {cv_ptr = _cv_ptr;_is_cv_linked=true;}
	void set_input_dim(unsigned _input_dim) {input_dim=_input_dim;}
	void set_output_dim(unsigned _output_dim) {output_dim=_output_dim;}
	
	void load_parameter(const std::string& filename);
	void save_parameter(const std::string& filename);

	unsigned get_input_dim() const {return input_dim;}
	unsigned get_output_dim() const {return output_dim;}
	
	dynet::ParameterCollection& get_model() {return pc;}
	
	Action* get_cv_ptr() const {
		plumed_massert(_is_cv_linked,"the action has not been linked");
		return cv_ptr;
	}
	
	virtual void clip(float left,float right,bool clip_last=false){}
	virtual void clip_inplace(float left,float right,bool clip_last=false){}
	
	virtual void build_neural_network(){}
	virtual dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x){return dynet::zeros(cg,{1});}
};

}

#endif
#endif
