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

#ifndef __PLUMED_bias_Deep_MetaD_h
#define __PLUMED_bias_Deep_MetaD_h

#include <random>
#include "Bias.h"
#include "tools/DynetTools.h"

namespace PLMD{
namespace bias{
	
using namespace dytools;

class Deep_MetaD: public Bias
{
private:
	unsigned narg;
	unsigned nlv;
	unsigned nlf;
	unsigned dynet_random_seed;
	unsigned update_steps;
	unsigned steps;
	unsigned update_cycle;
	unsigned mc_points;
	unsigned tot_mc_points;
	unsigned tot_md_points;
	unsigned sw_mc_points;
	unsigned sw_md_points;
	unsigned sc_mc_points;
	unsigned sc_md_points;
	unsigned hmc_steps;
	unsigned fes_nepoch;
	unsigned bias_nepoch;
	unsigned fes_bsize;
	unsigned bias_bsize;
	unsigned md_bsize;
	unsigned md_nepoch;
	unsigned mc_bsize;
	unsigned random_output_steps;
	unsigned mw_size;
	unsigned sw_size;
	unsigned ncores;
	
	bool firsttime;
	bool use_mw;
	bool use_mpi;
	bool no_update;
	bool is_arg_has_pbc;
	bool use_diff_param;
	bool use_hmc;
	bool use_same_bias_factor;
	bool clip_bias;
	bool clip_bias_last;
	bool clip_fes;
	bool clip_fes_last;
	
	float energy_scale;
	float scale_factor;
	float bias_factor;
	float mc_bias_factor;
	float bias_scale;
	//~ float mc_bias_scale;
	float dt_mc;
	
	float bias_clip_left;
	float bias_clip_right;
	float fes_clip_left;
	float fes_clip_right;
	float bias_clip_threshold;
	float fes_clip_threshold;
	
	double sim_temp;
	double beta;
	double kB;
	double kBT;
	
	std::default_random_engine rdgen;
	std::uniform_real_distribution<double> rand_prob;
	
	std::vector<bool> arg_pbc;
	//~ std::vector<unsigned> mc_seeds;
	std::vector<unsigned> md_ids;
	std::vector<unsigned> mc_ids;
	
	std::vector<float> lrv;
	std::vector<float> lrf;
	std::vector<float> hpv;
	std::vector<float> hpf;
	std::vector<float> mc_arg_sd;
	std::vector<float> hmc_arg_mass;
	
	std::vector<float> bias_record;
	//~ std::vector<float> weight_record;
	std::vector<float> fes_random;
	std::vector<float> bias_zero_args;
	std::vector<float> fes_zero_args;
	std::vector<float> arg_init;
	std::vector<float> arg_min;
	std::vector<float> arg_max;
	std::vector<float> arg_period;
	
	std::vector<std::string> arg_label;
	
	std::vector<float> arg_record;
	std::vector<float> arg_random;
	
	std::vector<float> mc_start_args;
	
	std::vector<std::normal_distribution<float>> ndist;

	std::string bias_file_in;
	std::string bias_file_out;
	std::string bias_algorithm;
	std::string fes_file_in;
	std::string fes_file_out;
	std::string fes_algorithm;
	std::string random_file;

	std::vector<unsigned> ldv;
	std::vector<unsigned> ldf;
	std::vector<Activation> afv;
	std::vector<Activation> aff;
	
	std::vector<std::vector<dynet::Parameter>> bias_params;
	std::vector<std::vector<dynet::Parameter>> fes_params;
	
	OFile orandom;
	
	Value* value_fes;
	Value* value_bloss;
	Value* value_floss;

	dynet::ParameterCollection pcv;
	dynet::ParameterCollection pcf;
	
	MLP_energy nnv;
	MLP_energy nnf;
	
	dynet::Trainer *trainer_bias;
	dynet::Trainer *trainer_fes;
	
	double random_velocities(std::vector<float>& v);
	void random_moves(std::vector<float>& coords);

public:
	explicit Deep_MetaD(const ActionOptions&);
	~Deep_MetaD();
	void calculate();
	void prepare();
	static void registerKeywords(Keywords& keys);
	
	unsigned get_random_seed() const {return dynet_random_seed;}
	void hybrid_monte_carlo(unsigned cycles);
	void metropolis_monte_carlo(unsigned cycles);
	
	float update_fes(std::vector<float>& fes_update);
	float update_bias(const std::vector<float>& fes_update);
	
	float calc_bias(const std::vector<float>& cvs,std::vector<float>& deriv){
		return nnv.calc_energy_and_deriv(cvs,deriv);
	}
	float calc_fes(const std::vector<float>& cvs,std::vector<float>& deriv){
		return nnf.calc_energy_and_deriv(cvs,deriv);
	}

	//~ void set_parameters();
};

}
}

#endif
#endif
