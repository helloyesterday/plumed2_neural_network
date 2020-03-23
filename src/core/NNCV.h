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

#ifndef __PLUMED_NNCV_h
#define __PLUMED_NNCV_h

#include "Action.h"
#include "Neural_Network.h"

namespace PLMD {

class NNCV :
	public virtual Action
{
protected:
	bool _is_nn_linked;
	bool _is_bias_linked;
	bool _is_calc_deriv;
	bool use_args;
	bool use_atoms;
	unsigned ninput;		// number of input argumnets
	unsigned input_dim;		// dimension of the input of neural network (equal or lager than ninput)
	unsigned output_dim;	// dimension of the output of neural network
	std::string nn_label;
	std::string param_file;
	
	Neural_Network* nn_ptr;
	Action* bias_ptr;
public:
	static void registerKeywords(Keywords&);
	explicit NNCV(const ActionOptions&ao);
	
	Neural_Network* get_nn_ptr() {return nn_ptr;}
	dynet::ParameterCollection& get_nn_model() {return nn_ptr->get_model();}
	Action* get_bais_ptr() {plumed_massert(_is_bias_linked,"the action has not been linked");return bias_ptr;}
	
	virtual void calculate() {}
	
	bool is_nn_linked() const {return _is_nn_linked;}
	bool is_bias_linked() const {return _is_bias_linked;}
	bool use_argumnets_input() const {return use_args;}
	bool use_atoms_input() const {return use_atoms;}
	bool is_calc_deriv() const {return _is_calc_deriv;}
	
	void set_input_dim() {nn_ptr->set_input_dim(input_dim);}
	void set_input_dim(unsigned _input_dim) {input_dim=_input_dim;nn_ptr->set_input_dim(input_dim);}
	void set_output_dim() {nn_ptr->set_output_dim(output_dim);}
	void set_output_dim(unsigned _output_dim) {output_dim=_output_dim;nn_ptr->set_output_dim(output_dim);}
	
	void linkBias(Action* _bias_ptr) {bias_ptr = _bias_ptr;_is_bias_linked=true;}
	
	void build_neural_network(){nn_ptr->build_neural_network();}
	
	void save_parameter(const std::string& filename){nn_ptr->save_parameter(filename);}
	void load_parameter(const std::string& filename){nn_ptr->load_parameter(filename);}
	void load_parameter(){nn_ptr->load_parameter(param_file);}
	
	virtual std::vector<float> get_input_values() {return std::vector<float>();}
	virtual std::vector<float> get_input_layer() {return std::vector<float>();}
	
	unsigned get_input_dim() const {return nn_ptr->get_input_dim();}
	unsigned get_output_dim() const {return nn_ptr->get_output_dim();}
	unsigned get_input_number() const {return ninput;}
	std::string get_nn_label() const {return nn_label;}
	std::string parameter_filename() const {return param_file;}
	
	void clip(float left,float right,bool clip_last=false) {nn_ptr->clip(left,right,clip_last);}
	void clip_inplace(float left,float right,bool clip_last=false) {nn_ptr->clip_inplace(left,right,clip_last);}
	
	dynet::Expression nn_output(dynet::ComputationGraph& cg,const dynet::Expression& x){return nn_ptr->output(cg,x);}
	virtual dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x) {return nn_ptr->output(cg,x);}
};

}

#endif
#endif
