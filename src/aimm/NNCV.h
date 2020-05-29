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

#include "core/Action.h"
#include "Neural_Network.h"

namespace PLMD {
namespace aimm {

class NNCV :
	public virtual Action
{
protected:
	bool _is_nn_linked;
	bool _is_bias_linked;
	bool _is_calc_deriv;
	bool use_args;
	bool use_atoms;
	bool is_first;
	unsigned step;
	unsigned param_wstride;
	unsigned grid_wstride;
	unsigned ninput;		// number of input original data
	unsigned ncvs;			// number of input argumnets
	unsigned input_dim;		// dimension of the input of neural network (equal or lager than ninput)
	unsigned output_dim;	// dimension of the output of neural network
	
	float grid_output_scale;
	float grid_output_shift;
	
	std::string nn_label;
	std::string param_in;
	std::string param_out;
	std::string grid_file;
	
	std::vector<unsigned> grid_bins;
	
	OFile ogrid;
	
	Neural_Network* nn_ptr;
	Action* bias_ptr;
public:
	static void registerKeywords(Keywords&);
	explicit NNCV(const ActionOptions&ao);
	~NNCV();
	
	Neural_Network* get_nn_ptr() {return nn_ptr;}
	dynet::ParameterCollection& get_nn_model() {return nn_ptr->get_model();}
	Action* get_bais_ptr() {plumed_massert(_is_bias_linked,"the action has not been linked");return bias_ptr;}
	
	virtual void calculate() {}
	void update();
	void prepare();
	
	bool is_nn_linked() const {return _is_nn_linked;}
	bool is_bias_linked() const {return _is_bias_linked;}
	bool use_argumnets_input() const {return use_args;}
	bool use_atoms_input() const {return use_atoms;}
	bool is_calc_deriv() const {return _is_calc_deriv;}
	
	float get_grid_output_scale() const {return grid_output_scale;}
	float get_grid_output_shift() const {return grid_output_shift;}
	void set_grid_output_scale(float scale) {grid_output_scale=scale;}
	void set_grid_output_shift(float shift) {grid_output_shift=shift;}
	
	void ogrid_init(const std::string& filename,bool enforce_suffix=false);
	void ogrid_rewind() {ogrid.rewind();}
	void ogrid_flush() {ogrid.flush();}
	
	void set_cvs_number(unsigned _ncvs) {ncvs=_ncvs;}
	void set_input_number(unsigned _ninput) {ninput=_ninput;}
	void set_input_dim() {nn_ptr->set_input_dim(input_dim);}
	void set_input_dim(unsigned _input_dim) {input_dim=_input_dim;nn_ptr->set_input_dim(input_dim);}
	void set_output_dim() {nn_ptr->set_output_dim(output_dim);}
	void set_output_dim(unsigned _output_dim) {output_dim=_output_dim;nn_ptr->set_output_dim(output_dim);}
	void set_gird_file(const std::string& new_file) {grid_file=new_file;}
	
	void linkBias(Action* _bias_ptr) {bias_ptr = _bias_ptr;_is_bias_linked=true;}
	
	void build_neural_network(){nn_ptr->build_neural_network();}
	
	void save_parameter(const std::string& filename){nn_ptr->save_parameter(filename);}
	void load_parameter(const std::string& filename){nn_ptr->load_parameter(filename);}
	void save_parameter(){nn_ptr->save_parameter(param_out);}
	void load_parameter(){nn_ptr->load_parameter(param_in);}
	
	void set_parameters(const std::vector<float>& new_param) {nn_ptr->set_parameters(new_param);}
	std::vector<float> get_parameters() const {return nn_ptr->get_parameters();}
	
	// Original input data
	virtual std::vector<float> get_input_data() {return std::vector<float>();}
	// Input cvs
	virtual std::vector<float> get_input_cvs() {return std::vector<float>();}
	// Input layer for neural network
	virtual std::vector<float> get_input_layer() {return std::vector<float>();}
	
	virtual void write_grid_file(){}
	
	unsigned parameters_number() const {return nn_ptr->parameters_number();}
	unsigned get_input_dim() const {return nn_ptr->get_input_dim();}
	unsigned get_output_dim() const {return nn_ptr->get_output_dim();}
	unsigned get_cvs_number() const {return ncvs;}
	unsigned get_input_number() const {return ninput;}
	unsigned get_grid_wstride() const {return grid_wstride;}
	
	std::string get_nn_label() const {return nn_label;}
	std::string get_gird_file() const {return grid_file;}
	std::string parameter_filename() const {return param_in;}
	
	void clip(float left,float right,bool clip_last=false) {nn_ptr->clip(left,right,clip_last);}
	void clip_inplace(float left,float right,bool clip_last=false) {nn_ptr->clip_inplace(left,right,clip_last);}
	
	virtual std::vector<bool> inputs_are_periodic() const {return std::vector<bool>();}
	virtual std::vector<float> inputs_min() const {return std::vector<float>();}
	virtual std::vector<float> inputs_max() const {return std::vector<float>();}
	virtual std::vector<float> inputs_period() const {return std::vector<float>();}
	
	virtual bool input_is_periodic(unsigned id) const {return false;}
	virtual float input_min(unsigned) const {return 0;}
	virtual float input_max(unsigned) const {return 0;}
	virtual float input_period(unsigned) const {return 0;}
	
	dynet::Expression nn_output(dynet::ComputationGraph& cg,const dynet::Expression& x){return nn_ptr->output(cg,x);}
	virtual dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x){return nn_ptr->output(cg,x);}
};

}
}

#endif
#endif
