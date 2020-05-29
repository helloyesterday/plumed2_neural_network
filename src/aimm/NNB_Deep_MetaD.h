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

#ifndef __PLUMED_bias_NNB_Deep_MetaD_h
#define __PLUMED_bias_NNB_Deep_MetaD_h

#include <random>
#include "NN_Bias.h"
#include "NNCV.h"
#include "Neural_Network.h"
#include "Trainer_Algorithm.h"

namespace PLMD{
namespace aimm{

class NNB_Deep_MetaD: public NN_Bias
{
private:
	bool firsttime;

	bool use_hmc;
	bool use_same_gamma;
	bool is_random_ouput;
	
	unsigned narg;
	unsigned step;
	unsigned update_steps;
	unsigned update_cycle;
	unsigned mc_points;
	unsigned tot_mc_points;
	unsigned tot_md_points;
	unsigned sw_mc_points;
	unsigned sw_md_points;
	unsigned sc_mc_points;
	unsigned sc_md_points;
	unsigned hmc_steps;
	unsigned fes_nbatch;
	unsigned bias_nbatch;
	unsigned fes_nepoch;
	unsigned bias_nepoch;
	unsigned fes_bsize;
	unsigned bias_bsize;
	unsigned md_bsize;
	unsigned md_nbatch;
	unsigned mc_bsize;
	unsigned random_output_steps;
	unsigned mw_size;
	unsigned sw_size;
	unsigned mw_rank;
	unsigned sw_rank;
	unsigned ncores;
	unsigned bias_output_steps;
	unsigned fes_output_steps;
	
	float bias_scale;
	float fes_scale;
	float wt_gamma;
	float mc_gamma;
	float dt_mc;
	float bias_loss_weight;
	float fes_loss_weight;
	float log_Z;
	float log_fes_bsize;
	float rct_scale;
	
	double bias_now;
	double rct;
	
	std::vector<unsigned> md_ids;
	std::vector<unsigned> mc_ids;
	
	std::vector<float> mc_arg_sd;
	std::vector<float> hmc_arg_mass;
	std::vector<float> bias_record;
	std::vector<float> fes_random;
	std::vector<float> fes_update;
	
	std::vector<float> args_now;
	std::vector<float> arg_record;
	std::vector<float> arg_random;
	std::vector<float> mc_start_args;

	std::string fes_nncv_label;
	std::string opt_bias_label;
	std::string opt_fes_label;
	std::string random_file;
	
	std::vector<std::string> arg_label;
	
	std::default_random_engine rdgen;
	std::uniform_real_distribution<double> rand_prob;
	std::vector<std::normal_distribution<float>> ndist;
	
	NNCV* nnv;
	NNCV* nnf;
	
	Trainer_Algorithm *algorithm_bias;
	Trainer_Algorithm *algorithm_fes;
	
	dynet::Trainer *trainer_bias;
	dynet::Trainer *trainer_fes;
	
	OFile orandom;
	
	Value* value_rct;
	Value* value_rbias;
	Value* value_fes;
	Value* value_Vloss;
	Value* value_Floss;
	
	double random_velocities(std::vector<float>& v);
	void random_moves(std::vector<float>& coords);
	//~ void write_grid_file(MLP_energy& nn,OFile& ogrid,const std::string& label);
	
	float calc_energy(dynet::ComputationGraph& cg,const std::vector<float>& cvs,NNCV* nncv);
	float calc_energy(const std::vector<float>& cvs,NNCV* nncv);
	float calc_energy_and_deriv(dynet::ComputationGraph& cg,const std::vector<float>& cvs,std::vector<float>& deriv,NNCV* nncv);
	float calc_energy_and_deriv(const std::vector<float>& cvs,std::vector<float>& deriv,NNCV* nncv);

public:
	explicit NNB_Deep_MetaD(const ActionOptions&);
	~NNB_Deep_MetaD();
	static void registerKeywords(Keywords& keys);
	void calculate();
	void update();
	
	void hybrid_monte_carlo(unsigned cycles);
	void metropolis_monte_carlo(unsigned cycles);
	
	float update_fes(const std::vector<float>& old_arg_record,
		const std::vector<float>& old_bias_record,
		const std::vector<float>& old_arg_random,
		const std::vector<float>& old_fes_random,
		std::vector<float>& new_arg_record,
		std::vector<float>& new_bias_record,
		std::vector<float>& new_arg_random,
		std::vector<float>& new_fes_random,
		std::vector<float>& update_fes_random);
	float update_bias(const std::vector<float>& vec_arg_record,
		const std::vector<float>& vec_bias_record,
		const std::vector<float>& vec_arg_random,
		const std::vector<float>& origin_fes_random,
		const std::vector<float>& update_fes_random);
		
	void fes_trainer_update(){trainer_fes->update();algorithm_fes->nn_clip(nnf->get_nn_ptr());}
	void bias_trainer_update(){trainer_bias->update();algorithm_bias->nn_clip(nnv->get_nn_ptr());}
	
	dynet::Expression nnv_output (dynet::ComputationGraph& cg,const dynet::Expression& x){return nnv->output(cg,x);}
	dynet::Expression nnf_output (dynet::ComputationGraph& cg,const dynet::Expression& x){return nnf->output(cg,x);}
	
	dynet::Expression fes_output (dynet::ComputationGraph& cg,const dynet::Expression& x){return nnf_output(cg,x)*fes_scale;}
	
	void set_bias_param(const std::vector<float>& new_param) {nnv->set_parameters(new_param);}
	std::vector<float> get_bias_param() const {return nnv->get_parameters();}
	void set_fes_param(const std::vector<float>& new_param) {nnf->set_parameters(new_param);}
	std::vector<float> get_fes_param() const {return nnf->get_parameters();}
	
	bool arg_is_periodic(unsigned id) const {return nncv_ptr->input_is_periodic(id);}
	float arg_min(unsigned id) const {return nncv_ptr->input_min(id);}
	float arg_max(unsigned id) const {return nncv_ptr->input_max(id);}
	float arg_period(unsigned id) const {return nncv_ptr->input_period(id);}
		
	float calc_bias(dynet::ComputationGraph& cg,const std::vector<float>& cvs)
		{return calc_energy(cg,cvs,nnv);}
	float calc_bias(const std::vector<float>& cvs) {return calc_energy(cvs,nnv);}
	float calc_bias_and_deriv(dynet::ComputationGraph& cg,const std::vector<float>& cvs,std::vector<float>& deriv)
		{return calc_energy_and_deriv(cg,cvs,deriv,nnv);}
	float calc_bias_and_deriv(const std::vector<float>& cvs,std::vector<float>& deriv)
		{return calc_energy_and_deriv(cvs,deriv,nnv);}
		
	float calc_fes(dynet::ComputationGraph& cg,const std::vector<float>& cvs)
		{return calc_energy(cg,cvs,nnf);}
	float calc_fes(const std::vector<float>& cvs) {return calc_energy(cvs,nnf);}
	float calc_fes_and_deriv(dynet::ComputationGraph& cg,const std::vector<float>& cvs,std::vector<float>& deriv)
		{return calc_energy_and_deriv(cg,cvs,deriv,nnf);}
	float calc_fes_and_deriv(const std::vector<float>& cvs,std::vector<float>& deriv)
		{return calc_energy_and_deriv(cvs,deriv,nnf);}

};

}
}

#endif
#endif
