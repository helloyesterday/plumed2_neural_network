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
	unsigned random_seed;
	unsigned update_steps;
	unsigned steps;
	unsigned hmc_points;
	unsigned tot_hmc_points;
	unsigned hmc_steps;
	unsigned fes_nepoch;
	unsigned bias_nepoch;
	unsigned fes_bsize;
	unsigned bias_bsize;
	unsigned md_bsize;
	unsigned md_nepoch;
	unsigned mc_bsize;
	
	bool firsttime;
	bool use_mw;
	bool no_update;
	bool is_arg_has_pbc;
	bool use_diff_param;
	
	float energy_scale;
	float scale_factor;
	float bias_factor;
	float bias_scale;
	float dt_hmc;
	
	float clip_left;
	float clip_right;
	float bias_clip_threshold;
	float fes_clip_threshold;
	
	double sim_temp;
	double beta;
	double kB;
	double kBT;
	
	std::default_random_engine rdgen;
	std::uniform_real_distribution<double> rand_prob;
	
	std::vector<bool> arg_pbc;
	
	std::vector<float> lrv;
	std::vector<float> lrf;
	std::vector<float> hpv;
	std::vector<float> hpf;
	std::vector<float> hmc_arg_mass;
	std::vector<float> hmc_arg_sd;
	
	std::vector<float> bias_record;
	std::vector<float> weight_record;
	std::vector<float> fes_random;
	std::vector<float> zero_args;
	std::vector<float> bias_zero_args;
	std::vector<float> fes_zero_args;
	std::vector<float> arg_init;
	std::vector<float> arg_min;
	std::vector<float> arg_max;
	std::vector<float> arg_period;
	
	std::vector<std::vector<float>> arg_record;
	std::vector<std::vector<float>> arg_random;
	
	std::vector<std::normal_distribution<float>> ndist;

	std::string bias_file_in;
	std::string bias_file_out;
	std::string bias_algorithm;
	std::string fes_file_in;
	std::string fes_file_out;
	std::string fes_algorithm;

	std::vector<unsigned> ldv;
	std::vector<unsigned> ldf;
	std::vector<Activation> afv;
	std::vector<Activation> aff;
	
	std::vector<std::vector<dynet::Parameter>> bias_params;
	std::vector<std::vector<dynet::Parameter>> fes_params;
	
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
	float get_output_and_gradient(dynet::ComputationGraph& cg,dynet::Expression& inputs,dynet::Expression& output,std::vector<float>& deriv);
	
public:
	explicit Deep_MetaD(const ActionOptions&);
	~Deep_MetaD();
	void calculate();
	void prepare();
	static void registerKeywords(Keywords& keys);
	
	float update_fes(std::vector<float>& fes_update);
	float update_bias(const std::vector<float>& fes_update);
	
	unsigned get_random_seed() const {return random_seed;}
	void hybrid_monte_carlo(std::vector<float>& init_coords);
	float calc_bias(const std::vector<float>& args,std::vector<float>& deriv);
	float calc_fes(const std::vector<float>& args,std::vector<float>& deriv);

	//~ void set_parameters();
};

}
}

#endif
#endif
