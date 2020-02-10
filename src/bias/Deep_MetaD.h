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
	unsigned hmc_steps;
	
	bool firsttime;
	bool use_mw;
	bool no_update;
	
	float bias_scale;
	float scale_factor;
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
	
	std::vector<float> lrv;
	std::vector<float> lrf;
	std::vector<float> hpv;
	std::vector<float> hpf;
	std::vector<float> hmc_arg_mass;
	std::vector<float> hmc_arg_sd;
	
	std::vector<float> arg_record;
	std::vector<float> arg_random;
	
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

	dynet::ParameterCollection pcv;
	dynet::ParameterCollection pcf;
	
	MLP nnv;
	MLP nnf;
	
	dynet::Trainer *trainer_bias;
	dynet::Trainer *trainer_fes;
	
	float calc_energy(const std::vector<float>& args,std::vector<float>& deriv);
	double random_velocities(std::vector<float>& v);
	double get_output_and_gradient(dynet::ComputationGraph& cg,dynet::Expression& inputs,dynet::Expression& output,std::vector<float>& deriv);
	
public:
	explicit Deep_MetaD(const ActionOptions&);
	~Deep_MetaD();
	void calculate();
	void prepare();
	static void registerKeywords(Keywords& keys);
	
	float update_fes();
	float update_bias();
	
	unsigned get_random_seed() const {return random_seed;}
	std::vector<float> hybrid_monte_carlo(const std::vector<float>& init_coords);

	//~ void set_parameters();
};

}
}

#endif
#endif
