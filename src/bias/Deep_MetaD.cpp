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

#include "Deep_MetaD.h"
#include "ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/Exception.h"
#include "tools/Matrix.h"
#include "tools/Random.h"
#include "tools/Tools.h"
#include <cstring>
#include "tools/File.h"
#include <iostream>
#include <iomanip>
#include <ctime>

namespace PLMD{
namespace bias{
	
using namespace dytools;

//+PLUMEDOC BIAS DEEP_METAD
/*

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(Deep_MetaD,"DEEP_METAD")

void Deep_MetaD::registerKeywords(Keywords& keys)
{
	Bias::registerKeywords(keys);
	ActionWithValue::useCustomisableComponents(keys);
	keys.addOutputComponent("fes","default","loss function for the bias potential (political network)");
	keys.addOutputComponent("bloss","default","loss function for the bias potential (political network)");
	keys.addOutputComponent("floss","default","loss function of the free energy surface (value network)");
	keys.remove("ARG");
    keys.add("optional","ARG","the argument(CVs) here must be the potential energy. If no argument is setup, only bias will be integrated in this method."); 
	//~ keys.use("ARG");

	keys.add("compulsory","UPDATE_STEPS","1000","the frequency to update parameters of neural networks. ZERO means do not update the parameters");
	keys.add("compulsory","LAYERS_NUMBER","3","the hidden layers of the multilayer preceptron of the neural networks");
	keys.add("compulsory","LAYER_DIMENSIONS","64","the dimension of each hidden layers of the neural networks");
	keys.add("compulsory","ACTIVE_FUNCTIONS","SWISH","the activation function for the neural networks");
	keys.add("compulsory","OPT_ALGORITHM","ADAM","the algorithm to train the neural networks");
	keys.add("compulsory","SCALE_FACTOR","10","a constant to scale the output of neural network as bias potential");
	keys.add("compulsory","BIAS_PARAM_OUTPUT","bias_parameters.data","the file to output the parameters of the neural network of the bias potential");
	keys.add("compulsory","FES_PARAM_OUTPUT","fes_parameters.data","the file to output the parameters of the neural network of the free energy surface");
	keys.add("compulsory","BATCH_SIZE","250","batch size in each epoch for training neural networks");
	keys.add("compulsory","ZERO_REF_ARG","0,0","the reference arguments to make the energy of neural networks as zero");
	keys.add("compulsory","CLIP_RANGE","0.1,0.1","the range of the value to clip");
	keys.add("compulsory","MC_SAMPLING_POINTS","100","the sampling points for hybrid monte carlo at a MD simulation step");
	keys.add("compulsory","MC_TIMESTEP","0.001","the time step for hybrid monte carlo for the sampling of the neural network of free energy");
	keys.add("compulsory","MC_ARG_STDEV","1.0","the standard deviation of the normal distribution to generate random velocity for each argument at hybrid monte carlo");
	keys.add("compulsory","HMC_STEPS","10","the steps for a sampling of a hybrid monte carlo run");
	keys.add("compulsory","HMC_ARG_MASS","1.0","the mass for each argument at hybrid monte carlo");
	keys.add("compulsory","WT_BIAS_FACTOR","10","the well-tempered bias factor for deep metadynamics");
	
	keys.addFlag("USE_HYBRID_MONTE_CARLO",false,"use hybrid monte carlo to sample the neural networks");
	keys.addFlag("USE_DIFF_PARAM_FOR_FES",false,"use different parameters for the neural network of free energy surface");
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
	keys.addFlag("NO_BIAS_CLIP",false,"do not clip the neural network of bias potential");
	keys.addFlag("NO_FES_CLIP",false,"do not clip the neural network of free energy");
	keys.addFlag("CLIP_BIAS_LAST_LAYER",false,"clip the last layer of the neural network of bias potential");
	keys.addFlag("CLIP_FES_LAST_LAYER",false,"clip the last layer of the neural network of free energy");
	
	keys.add("optional","RANDOM_POINTS_FILE","the file to output the random points of monte carlo");
	keys.add("optional","RANDOM_OUTPUT_STEPS","(default=1) the frequency to output the random points file");
	keys.add("optional","MC_BIAS_FACTOR","the well-tempered bias factor for monte carlo");
	keys.add("optional","LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","HYPER_PARAMS","other hyperparameters for training the neural network");
	keys.add("optional","CLIP_THRESHOLD","the clip threshold for training the neural network");
	keys.add("optional","FES_CLIP_RANGE","the range of the value to clip");
	keys.add("optional","FES_ZERO_REF_ARG","the reference arguments to make the free energy as zero");
	keys.add("optional","FES_BATCH_SIZE","batch size in each epoch for training free energy surface");
	keys.add("optional","FES_LAYERS_NUMBER","the hidden layers of the neural network for the FES");
	keys.add("optional","FES_LAYER_DIMENSIONS","the dimension of each hidden layers of the neural network for the FES");
	keys.add("optional","FES_ACTIVE_FUNCTIONS","the activation function of the neural network for the FES");
	keys.add("optional","FES_OPT_ALGORITHM","the algorithm to train the discriminator (free energy surface)");
	keys.add("optional","FES_LEARN_RATE","the learning rate for the neural network of FES");
	keys.add("optional","FES_HYPER_PARAMS","other hyperparameters for training the neural network of FES");
	keys.add("optional","FES_CLIP_THRESHOLD","the clip threshold for training the neural network of FES");
	keys.add("optional","BIAS_PARAM_READ_FILE","the file to output the parameters of the neural network of bias potential");
	keys.add("optional","FES_PARAM_READ_FILE","the file to output the parameters of the neural networkof free energy surface");
	keys.add("optional","SIM_TEMP","the simulation temerature");
	
}

Deep_MetaD::Deep_MetaD(const ActionOptions& ao):
	PLUMED_BIAS_INIT(ao),
	narg(getNumberOfArguments()),
	steps(0),
	update_cycle(0),
	firsttime(true),
	rand_prob(0,1.0),
	arg_pbc(getNumberOfArguments(),false),
	arg_init(getNumberOfArguments(),0),
	arg_min(getNumberOfArguments(),0),
	arg_max(getNumberOfArguments(),0),
	arg_period(getNumberOfArguments(),0),
	trainer_bias(NULL),
	trainer_fes(NULL)
{
	is_arg_has_pbc=false;
	for(unsigned i=0;i!=narg;++i)
	{
		arg_pbc[i]=getPntrToArgument(i)->isPeriodic();
		arg_label.push_back(getPntrToArgument(i)->getName());
		if(arg_pbc[i])
			is_arg_has_pbc=true;

		double minout,maxout;
		getPntrToArgument(i)->getDomain(minout,maxout);
		arg_min[i]=minout;
		arg_max[i]=maxout;
		arg_period[i]=maxout-minout;
	}
	nnv.set_cvs_number(narg);
	nnv.set_periodic(arg_pbc);
	nnf.set_cvs_number(narg);
	nnf.set_periodic(arg_pbc);
	
	parseFlag("USE_DIFF_PARAM_FOR_FES",use_diff_param);
	
	parse("LAYERS_NUMBER",nlv);
	plumed_massert(nlv>0,"LAYERS_NUMBER must be larger than 0!");
	nlf=nlv;

	parseVector("LAYER_DIMENSIONS",ldv);
	if(ldv.size()!=nlv)
	{
		if(ldv.size()==1)
		{
			unsigned dim=ldv[0];
			ldv.assign(nlv,dim);
		}
		else
			plumed_merror("the size of LAYER_DIMENSIONS mismatch!");
	}
	ldf=ldv;

	std::vector<std::string> safv;
	parseVector("ACTIVE_FUNCTIONS",safv);
	if(safv.size()!=nlv)
	{
		if(safv.size()==1)
		{
			std::string af=safv[0];
			safv.assign(nlv,af);
		}
		else
			plumed_merror("the size of ACTIVE_FUNCTIONS mismatch!");
	}
	std::vector<std::string> saff=safv;
	
	parseVector("ZERO_REF_ARG",bias_zero_args);
	if(bias_zero_args.size()!=narg)
	{
		if(bias_zero_args.size()==1)
		{
			float val=bias_zero_args[0];
			bias_zero_args.assign(narg,val);
		}
		else
			plumed_merror("the size of ZERO_REF_ARG mismatch!");
	}
	fes_zero_args=bias_zero_args;
	
	afv.resize(nlv);
	for(unsigned i=0;i!=nlv;++i)
		afv[i]=activation_function(safv[i]);
	aff=afv;
	
	if(use_diff_param)
	{
		parse("FES_LAYERS_NUMBER",nlf);
		plumed_massert(nlv>0,"LAYERS_NUMBER must be larger than 0!");
		std::vector<unsigned> _ldf;
		parseVector("FES_LAYER_DIMENSIONS",_ldf);
		if(_ldf.size()>0)
			ldf=_ldf;
		if(ldf.size()!=nlf)
		{
			if(ldf.size()==1)
			{
				unsigned dim=ldf[0];
				ldf.assign(nlf,dim);
			}
			else
				plumed_merror("FES_LAYER_DIMENSIONS mismatch, nlf and ldf.size(): "+std::to_string(nlf)+" and "+std::to_string(ldf.size()));
		}
		
		std::vector<std::string> _saff;
		parseVector("FES_ACTIVE_FUNCTIONS",_saff);
		if(_saff.size()>0)
			saff=_saff;
		if(saff.size()>0)
		{
			if(saff.size()!=nlf)
			{
				if(nlf>0&&saff.size()==0)
					saff=safv;
				if(saff.size()==1)
				{
					std::string af=saff[0];
					saff.assign(nlf,af);
				}
				else
					plumed_merror("FES_ACTIVE_FUNCTIONS mismatch!");
			}
			aff.resize(nlf);
			for(unsigned i=0;i!=nlf;++i)
				aff[i]=activation_function(saff[i]);
		}
	}

	parse("BIAS_PARAM_OUTPUT",bias_file_out);
	parse("FES_PARAM_OUTPUT",fes_file_out);
	parse("BIAS_PARAM_READ_FILE",bias_file_in);
	parse("FES_PARAM_READ_FILE",fes_file_in);
	parse("RANDOM_POINTS_FILE",random_file);
	random_output_steps=1;
	parse("RANDOM_OUTPUT_STEPS",random_output_steps);
	
	parse("SCALE_FACTOR",scale_factor);
	parse("WT_BIAS_FACTOR",bias_factor);
	plumed_massert(bias_factor>0,"WT_BIAS_FACTOR shoud larger than 0");
	mc_bias_factor=bias_factor;
	bias_scale=0;
	
	use_same_bias_factor=true;
	float _mc_bias_factor=-1;
	parse("MC_BIAS_FACTOR",_mc_bias_factor);
	if(mc_bias_factor>0)
	{
		use_same_bias_factor=false;
		mc_bias_factor=_mc_bias_factor;
		bias_scale=1.0/mc_bias_factor-1.0/bias_factor;
	}
	//~ mc_bias_scale=1.0/mc_bias_factor-1.0;
	
	kB=plumed.getAtoms().getKBoltzmann();
	sim_temp=-1;
	parse("SIM_TEMP",sim_temp);
	if(sim_temp>0)
		kBT=kB*sim_temp;
	else
	{
		kBT=plumed.getAtoms().getKbT();
		sim_temp=kBT/kB;
	}
	beta=1.0/kBT;
	
	energy_scale=scale_factor*kBT;
	nnv.set_energy_scale(energy_scale);
	nnf.set_energy_scale(energy_scale);
	
	parseFlag("MULTIPLE_WALKERS",use_mw);

	random_seed=0;
	if(use_mw)
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				std::random_device rd;
				random_seed=rd();
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(random_seed,0);
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			std::random_device rd;
			random_seed=rd();
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);
	}

	dynet_initialization(random_seed);
		
	std::string bias_fullname;
	std::string fes_fullname;
	
	parse("UPDATE_STEPS",update_steps);
	parse("BATCH_SIZE",bias_bsize);
	if(update_steps>0)
		plumed_massert(update_steps%bias_bsize==0,"UPDATE_STEPS must be divided exactly by BATCH_SIZE");
	fes_bsize=bias_bsize;
	
	std::vector<float> bclips;
	parseVector("CLIP_RANGE",bclips);
	plumed_massert(bclips.size()>=2,"CLIP_RANGE should has left and right values");
	plumed_massert(bclips[1]>bclips[0],"Clip left value should less than clip right value");
	bias_clip_left=bclips[0];
	bias_clip_right=bclips[1];
	fes_clip_left=bias_clip_left;
	fes_clip_right=bias_clip_right;
	
	parseFlag("USE_HYBRID_MONTE_CARLO",use_hmc);
	
	bool no_bias_clip;
	parseFlag("NO_BIAS_CLIP",no_bias_clip);
	clip_bias=!no_bias_clip;
	bool no_fes_clip;
	parseFlag("NO_FES_CLIP",no_fes_clip);
	clip_fes=!no_fes_clip;
	
	parseFlag("CLIP_BIAS_LAST_LAYER",clip_bias_last);
	parseFlag("CLIP_FES_LAST_LAYER",clip_fes_last);
	
	parse("MC_SAMPLING_POINTS",mc_points);
	parse("MC_TIMESTEP",dt_mc);
	parseVector("MC_ARG_STDEV",mc_arg_sd);
	
	if(mc_arg_sd.size()!=narg)
	{
		if(mc_arg_sd.size()==1)
		{
			float hsd=mc_arg_sd[0];
			mc_arg_sd.assign(narg,hsd);
		}
		else
			plumed_merror("the number of HMC_ARG_STDEV mismatch");
	}
	for(unsigned i=0;i!=narg;++i)
		ndist.push_back(std::normal_distribution<float>(0,mc_arg_sd[i]));

	parse("HMC_STEPS",hmc_steps);
	parseVector("HMC_ARG_MASS",hmc_arg_mass);
	if(hmc_arg_mass.size()!=narg)
	{
		if(hmc_arg_mass.size()==1)
		{
			float hmass=hmc_arg_mass[0];
			hmc_arg_mass.assign(narg,hmass);
		}
		else
			plumed_merror("the number of HMC_ARG_MASS mismatch");
	}
		
	parse("OPT_ALGORITHM",bias_algorithm);
	fes_algorithm=bias_algorithm;
	
	parseVector("LEARN_RATE",lrv);
	lrf=lrv;
	std::vector<float> opv;
	parseVector("HYPER_PARAMS",opv);
	std::vector<float> opf=opv;
	
	if(update_steps>0)
	{
		tot_mc_points=update_steps*mc_points;
		bias_nepoch=update_steps/bias_bsize;
		fes_nepoch=tot_mc_points/fes_bsize;
		mc_bsize=tot_mc_points/bias_nepoch;
		
		if(random_file.size()>0)
		{
			orandom.link(*this);
			orandom.open(random_file.c_str());
			orandom.fmtField(" %f");
			orandom.addConstantField("ITERATION");
		}
		
		if(use_diff_param)
		{
			parse("FES_OPT_ALGORITHM",fes_algorithm);
			
			std::vector<float> _lrf;
			parseVector("FES_LEARN_RATE",_lrf);
			if(_lrf.size()>0)
				lrf=_lrf;
				
			std::vector<float> _opf;
			parseVector("FES_HYPER_PARAMS",_opf);
			if(_opf.size()>0)
				opf=_opf;
			
			parse("FES_BATCH_SIZE",fes_bsize);
			plumed_massert(tot_mc_points%fes_bsize==0,"the total sampling points of HMC must be divided exactly by FES_BATCH_SIZE");
			
			std::vector<float> _fes_zero_args;
			parseVector("FES_ZERO_REF_ARG",_fes_zero_args);
			if(_fes_zero_args.size()>0)
				fes_zero_args=_fes_zero_args;
			if(fes_zero_args.size()!=narg)
			{
				if(fes_zero_args.size()==1)
				{
					float val=fes_zero_args[0];
					fes_zero_args.assign(narg,val);
				}
				else
					plumed_merror("the size of FES_ZERO_REF_ARG mismatch!");
			}
			
			std::vector<float> fclips;
			parseVector("FES_CLIP_RANGE",fclips);
			if(fclips.size()>0)
			{
				plumed_massert(fclips.size()>=2,"FES_CLIP_RANGE should has left and right values");
				plumed_massert(fclips[1]>fclips[0],"Clip left value should less than clip right value (FES_CLIP_RANGE)");
				fes_clip_left=fclips[0];
				fes_clip_right=fclips[1];
			}
		}
		
		nnv.set_zero_cvs(bias_zero_args);
		nnf.set_zero_cvs(fes_zero_args);
		
		md_bsize=fes_bsize;
		if(fes_bsize>update_steps)
			md_bsize=update_steps;
		md_nepoch=update_steps/md_bsize;
		
		if(lrv.size()==0)
		{
			trainer_bias=new_traniner(bias_algorithm,pcv,bias_fullname);
		}
		else
		{
			if(bias_algorithm=="CyclicalSGD"||bias_algorithm=="cyclicalSGD"||bias_algorithm=="cyclicalsgd"||bias_algorithm=="CSGD"||bias_algorithm=="csgd")
			{
				plumed_massert(lrv.size()==2,"The CyclicalSGD algorithm need two learning rates");
			}
			else
			{
				plumed_massert(lrv.size()==1,"The "+bias_algorithm+" algorithm need only one learning rates");
			}
			
			hpv.insert(hpv.end(),lrv.begin(),lrv.end());
			
			if(opv.size()>0)
				hpv.insert(hpv.end(),opv.begin(),opv.end());
			
			trainer_bias=new_traniner(bias_algorithm,pcv,hpv,bias_fullname);
		}
		
		if(lrf.size()==0)
		{
			trainer_fes=new_traniner(fes_algorithm,pcf,fes_fullname);
		}
		else
		{
			if(fes_algorithm=="CyclicalSGD"||fes_algorithm=="cyclicalSGD"||fes_algorithm=="cyclicalsgd"||fes_algorithm=="CSGD"||fes_algorithm=="csgd")
			{
				plumed_massert(lrf.size()==2,"The CyclicalSGD algorithm need two learning rates");
			}
			else
			{
				plumed_massert(lrf.size()==1,"The "+fes_algorithm+" algorithm need only one learning rates");
			}
			
			hpf.insert(hpf.end(),lrf.begin(),lrf.end());
			
			if(opf.size()>0)
				hpf.insert(hpf.end(),opf.begin(),opf.end());
			
			trainer_fes=new_traniner(fes_algorithm,pcf,hpf,fes_fullname);
		}
		
		bias_clip_threshold=trainer_bias->clip_threshold;
		parse("CLIP_THRESHOLD",bias_clip_threshold);
		fes_clip_threshold=bias_clip_threshold;
		parse("FES_CLIP_THRESHOLD",fes_clip_threshold);
		
		trainer_bias->clip_threshold = bias_clip_threshold;
		trainer_fes->clip_threshold = fes_clip_threshold;
	}
	
	if(update_steps>0)
	{
		addComponent("fes"); componentIsNotPeriodic("fes");
		value_fes=getPntrToComponent("fes");
		addComponent("bloss"); componentIsNotPeriodic("bloss");
		value_bloss=getPntrToComponent("bloss");
		addComponent("floss"); componentIsNotPeriodic("floss");
		value_floss=getPntrToComponent("floss");
	}
	
	checkRead();

	for(unsigned i=0;i!=narg;++i)
	{
		if(arg_pbc[i])
			log.printf("    %dth CV \"%s\" (periodic): from %f to %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i]);
		else
			log.printf("    %dth CV \"%s\" (non-periodic): from %f to %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i]);
	}
	if(use_mw)
		log.printf("  using multiple walkers: \n");
	log.printf("  with random seed: %d\n",int(random_seed));
	log.printf("  with number of input argument: %d\n",int(narg));
	
	log.printf("  with neural network for bias potential (political network):\n");
	log.printf("    with energy scale: %f\n",energy_scale);
	log.printf("    with number of hidden layers: %d\n",int(nlv));
	for(unsigned i=0;i!=nlv;++i)
		log.printf("      Hidden layer %d with dimension %d and activation funciton %s\n",int(i),int(ldv[i]),safv[i].c_str());
	log.printf("    with parameters output file: %s\n",bias_file_out.c_str());
	if(bias_file_in.size()>0)
		log.printf("    with parameters read file: %s\n",bias_file_in.c_str());
	
	if(update_steps>0)
	{
		log.printf("  update bias potential using deep metadynamics with well-tempered bias factor: %f.\n",bias_factor);
		log.printf("  with the neural network for free energy surface (value network):\n");
		log.printf("    with number of hidden layers: %d\n",int(nlf));
		for(unsigned i=0;i!=nlf;++i)
			log.printf("      Hidden layer %d with dimension %d and activation funciton %s\n",int(i),int(ldf[i]),saff[i].c_str());
		log.printf("    with parameters output file: %s\n",fes_file_out.c_str());
		if(fes_file_in.size()>0)
			log.printf("    with parameters read file: %s\n",fes_file_in.c_str());
		for(unsigned i=0;i!=narg;++i)
			log.printf("    %dth CV \"%s\" with reference value %f(bias) and %f(fes).\n",int(i+1),arg_label[i].c_str(),bias_zero_args[i],fes_zero_args[i]);
		
		if(use_hmc)
			log.printf("  using hybrid monte carlo for the sampling of free energy surface.\n");
		else
			log.printf("  using metropolis monte carlo for the sampling of free energy surface.\n");
		log.printf("    with well-tempered bias factor: %f.\n",mc_bias_factor);
		log.printf("    with sampling points for each MD step: %d.\n",int(mc_points));
		log.printf("    with time step: %f.\n",dt_mc);
		if(use_hmc)
		{
			log.printf("    with moving steps for each sampling point: %d.\n",int(hmc_steps));
			log.printf("    with total sampling points for each update cycle: %d.\n",int(tot_mc_points));
		}
		for(unsigned i=0;i!=narg;++i)
			log.printf("    %dth CV \"%s\" with standard deviation %f.\n",int(i+1),arg_label[i].c_str(),mc_arg_sd[i]);
		
		log.printf("  update the parameters of neural networks every %d steps.\n",int(update_steps));
		log.printf("  using optimization algorithm for the neural networks of bias potential (political network): %s.\n",bias_fullname.c_str());
		log.printf("    with batch size for traning bias potential at political networks: %d.\n",int(bias_bsize));
		log.printf("    with batch size for traning free energy surface at political networks: %d.\n",int(mc_bsize));
		if(clip_bias)
		{
			if(clip_bias_last)
				log.printf("    with clip range included last layer: from %f to %f.\n",bias_clip_left,bias_clip_right);
			else
				log.printf("    with clip range excluded last layer: from %f to %f.\n",bias_clip_left,bias_clip_right);
		}

		log.printf("  using optimization algorithm for the neural networks of free energy surface (value network): %s.\n",fes_fullname.c_str());
		log.printf("    with batch size for traning bias potential at value network: %d.\n",int(md_bsize));
		log.printf("    with batch size for traning free energy surface at value network: %d.\n",int(fes_bsize));
		if(clip_fes)
		{
			if(clip_fes_last)
				log.printf("    with clip range included last layer: from %f to %f.\n",fes_clip_left,fes_clip_right);
			else
				log.printf("    with clip range excluded last layer: from %f to %f.\n",fes_clip_left,fes_clip_right);
		}
	}
	else
		log.printf("    without updating the parameters of neural networks.\n");
		
}

Deep_MetaD::~Deep_MetaD()
{
	if(update_steps>0)
	{
		delete trainer_bias;
		delete trainer_fes;
		if(random_file.size()>0)
			orandom.close();
	}
}


void Deep_MetaD::prepare()
{
	if(firsttime)
	{
		nnv.set_hidden_layers(ldv,afv);
		nnv.build_neural_network(pcv);
		
		if(bias_file_in.size()>0)
		{
			dynet::TextFileLoader loader(bias_file_in);
			loader.populate(pcv);
		}
		dynet::TextFileSaver savev(bias_file_out);
		savev.save(pcv);
		
		if(update_steps>0)
		{
			//~ nnv.align_zero(bias_zero_args);
			
			nnf.set_hidden_layers(ldf,aff);
			nnf.build_neural_network(pcf);
			//~ nnf.align_zero(fes_zero_args);
			if(fes_file_in.size()>0)
			{
				dynet::TextFileLoader loader(fes_file_in);
				loader.populate(pcf);
			}
			dynet::TextFileSaver savef(fes_file_out);
			savef.save(pcf);
		}
		
		firsttime=false;
	}
}

void Deep_MetaD::calculate()
{
	std::vector<float> args(narg);
	for(unsigned i=0;i!=narg;++i)
	{
		float val=getArgument(i);
		args[i]=val;
	}

	std::vector<float> deriv;
	double bias_pot=calc_bias(args,deriv);
	
	setBias(bias_pot);
	
	for(unsigned i=0;i!=narg;++i)
	{
		double bias_force=-deriv[i];
		setOutputForce(i,bias_force);
	}
	
	if(update_steps>0)
	{
		double fes=calc_fes(args,deriv);
		value_fes->set(fes);
		
		bias_record.push_back(bias_pot);
		weight_record.push_back(std::exp(beta*bias_pot));
		arg_record.push_back(args);

		if(steps==0)
			arg_init=args;
		
		if(use_hmc)
			arg_init=hybrid_monte_carlo(arg_init);
		else
			arg_init=metropolis_monte_carlo(arg_init);

		if(steps>0&&(steps%update_steps==0))
		{
			std::vector<float> fes_update;
			double floss=update_fes(fes_update);
			value_floss->set(floss);
			double bloss=update_bias(fes_update);
			value_bloss->set(bloss);
			
			if(random_file.size()>0&&update_cycle%random_output_steps==0)
			{
				orandom.printField("ITERATION",getTime());
				for(unsigned i=0;i!=arg_random.size();++i)
				{
					orandom.printField("index",int(i));
					for(unsigned j=0;j!=narg;++j)
						orandom.printField(arg_label[j],arg_random[i][j]);
					orandom.printField("fes",fes_random[i]);
					orandom.printField();
				}
				orandom.flush();
			}
			
			bias_record.resize(0);
			weight_record.resize(0);
			arg_record.resize(0);
			fes_random.resize(0);
			arg_random.resize(0);
			arg_init.swap(args);
			++update_cycle;
		}
	}
	++steps;
}

float Deep_MetaD::update_fes(std::vector<float>& fes_update)
{
	//~ std::vector<float> weight_batch(md_bsize);
	std::vector<float> bias_batch(md_bsize);
	std::vector<float> fes_batch(fes_bsize);
	std::vector<float> arg_record_batch(narg*md_bsize);
	std::vector<float> arg_random_batch(narg*fes_bsize);

	if(fes_bsize>=update_steps)
	{
		//~ weight_batch=weight_record;
		bias_batch=bias_record;
		auto iter_md_batch=arg_record_batch.begin();
		for(unsigned i=0;i!=update_steps;++i)
			iter_md_batch=std::copy(arg_record[i].begin(),arg_record[i].end(),iter_md_batch);
	}

	dynet::ComputationGraph cg;

	//~ dynet::Dim weight_dim({1},md_bsize);
	//~ dynet::Dim fes_dim({1},fes_bsize);
	//~ dynet::Dim md_dim({narg},md_bsize);
	//~ dynet::Dim mc_dim({narg},fes_bsize);

	dynet::Dim bias_dim({md_bsize});
	dynet::Dim fes_dim({fes_bsize});
	dynet::Dim md_dim({narg,md_bsize});
	dynet::Dim mc_dim({narg,fes_bsize});

	dynet::Expression bias_inputs=dynet::input(cg,bias_dim,&bias_batch);
	dynet::Expression fes_inputs=dynet::input(cg,fes_dim,&fes_batch);
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record_batch);
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random_batch);
	dynet::Expression zero_inputs=dynet::input(cg,{narg},&fes_zero_args);

	dynet::Expression nn_md=nnf.MLP_output(cg,md_inputs);
	dynet::Expression nn_mc=nnf.MLP_output(cg,mc_inputs);
	//~ dynet::Expression fes_md=nnf.energy(cg,md_inputs);
	dynet::Expression fes_mc=nnf.energy(cg,mc_inputs);
	dynet::Expression fes_zero=nnf.MLP_output(cg,zero_inputs);

	//~ dynet::Expression md_weights=dynet::input(cg,weight_dim,&weight_batch);
	//~ dynet::Expression mc_weights=dynet::exp(beta*(fes_inputs/mc_bias_factor-fes_mc));
	
	dynet::Expression md_weights=dynet::softmax(bias_inputs*beta);
	dynet::Expression mc_weights=dynet::softmax((fes_inputs/mc_bias_factor-dynet::transpose(fes_mc))*beta);
	
	//~ dynet::Expression sum_sample=dynet::sum_batches(nn_md*md_weights);
	//~ dynet::Expression sum_sample_weight=dynet::sum_batches(md_weights);
	//~ dynet::Expression sum_target=dynet::sum_batches(nn_mc*md_weights);
	//~ dynet::Expression sum_target_weight=dynet::sum_batches(md_weights);
	
	//~ dynet::Expression mean_sample=sum_sample/sum_sample_weight;
	//~ dynet::Expression mean_target=sum_target/sum_target_weight;
	dynet::Expression mean_sample=nn_md*md_weights;
	dynet::Expression mean_target=nn_mc*mc_weights;
	
	dynet::Expression mean_zeros=0.5*dynet::squared_norm(fes_zero);

	dynet::Expression loss=mean_sample-mean_target+mean_zeros;

	unsigned seed=std::time(0);
	std::shuffle(fes_random.begin(),fes_random.end(),std::default_random_engine(seed));
	std::shuffle(arg_random.begin(),arg_random.end(),std::default_random_engine(seed));

	auto iter_fes=fes_random.begin();
	auto iter_mc=arg_random.begin();
	//~ auto iter_weight=weight_record.begin();
	auto iter_bias=bias_record.begin();
	auto iter_md=arg_record.begin();

	double floss=0;
	std::vector<float> vec_random_batch;
	for(unsigned i=0;i!=fes_nepoch;++i)
	{
		if(fes_bsize<update_steps&&(i%md_nepoch==0))
		{
			seed=std::time(0);
			std::shuffle(arg_record.begin(),arg_record.end(),std::default_random_engine(seed));
			//~ std::shuffle(weight_record.begin(),weight_record.end(),std::default_random_engine(seed));
			std::shuffle(bias_record.begin(),bias_record.end(),std::default_random_engine(seed));
			//~ iter_weight=weight_record.begin();
			iter_bias=bias_record.begin();
			iter_md=arg_record.begin();
		}
		auto iter_mc_batch=arg_random_batch.begin();
		auto iter_md_batch=arg_record_batch.begin();
		for(unsigned j=0;j!=fes_bsize;++j)
		{
			fes_batch[j]=*(iter_fes++);
			iter_mc_batch=std::copy(iter_mc->begin(),iter_mc->end(),iter_mc_batch);
			++iter_mc;
			if(fes_bsize<update_steps)
			{
				//~ weight_batch[j]=*(iter_weight++);
				bias_batch[j]=*(iter_bias++);
				iter_md_batch=std::copy(iter_md->begin(),iter_md->end(),iter_md_batch);
				++iter_md;
			}
		}
		std::copy(arg_random_batch.begin(),arg_random_batch.end(),std::back_inserter(vec_random_batch));
		floss += as_scalar(cg.forward(loss));
		cg.backward(loss);
		trainer_fes->update();
		if(clip_fes)
			nnf.clip_inplace(fes_clip_left,fes_clip_right,clip_fes_last);
		//~ nnf.update_energy_shift(cg);
		//~ nnf.align_zero(fes_zero_args);
	}
	floss/=fes_nepoch;
	
	dynet::Dim fin_dim({narg},tot_mc_points);
	dynet::Expression fin_inputs=dynet::input(cg,fin_dim,&vec_random_batch);
	dynet::Expression fin_output=nnf.energy(cg,fin_inputs);
	
	fes_update=dynet::as_vector(cg.forward(fin_output));
	
	return floss;
}

float Deep_MetaD::update_bias(const std::vector<float>& fes_update)
{
	dynet::ComputationGraph cg;
	
	std::vector<float> arg_record_batch(narg*bias_bsize);
	std::vector<float> arg_random_batch(narg*mc_bsize);
	std::vector<float> fes_batch(mc_bsize);
	std::vector<float> bias_batch(bias_bsize);
	
	//~ dynet::Dim md_dim({narg},bias_bsize);
	//~ dynet::Dim bias_dim({1},bias_bsize);
	dynet::Dim md_dim({narg,bias_bsize});
	dynet::Dim bias_dim({bias_bsize});
	
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record_batch);
	dynet::Expression bias_input=dynet::input(cg,bias_dim,&bias_batch);
	
	dynet::Expression nn_md=nnv.MLP_output(cg,md_inputs);
	//~ dynet::Expression bias_mc=nnv.energy(cg,mc_inputs);
	dynet::Expression bias_md=nnv.energy(cg,md_inputs);
	
	//~ dynet::Expression md_weights=dynet::exp(beta*(bias_input-bias_md));
	//~ dynet::Expression sum_sample=dynet::sum_batches(nn_md*md_weights);
	//~ dynet::Expression sum_sample_weight=dynet::sum_batches(md_weights);
	//~ dynet::Expression mean_sample=sum_sample/sum_sample_weight;
	
	dynet::Expression md_weights=dynet::softmax((bias_input-dynet::transpose(bias_md))*beta);
	dynet::Expression mean_sample=nn_md*md_weights;
	
	dynet::Dim mc_dim({narg,mc_bsize});
	
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random_batch);
	dynet::Expression nn_mc=nnv.MLP_output(cg,mc_inputs);
	
	dynet::Expression mean_target;
	if(use_same_bias_factor)
		mean_target=dynet::mean_elems(nn_mc);
	else
	{
		dynet::Dim fes_dim({1},mc_bsize);
		dynet::Expression fes_inputs=dynet::input(cg,fes_dim,&fes_batch);
		dynet::Expression mc_weights=dynet::softmax(beta*bias_scale*fes_inputs);
		mean_target=nn_mc*mc_weights;
	}
	
	dynet::Expression zero_inputs=dynet::input(cg,{narg},&bias_zero_args);
	dynet::Expression bias_zero=nnv.MLP_output(cg,zero_inputs);
	dynet::Expression mean_zeros=0.5*dynet::squared_norm(bias_zero);
	
	dynet::Expression loss=mean_target-mean_sample+mean_zeros;
	//~ dynet::Expression loss=mean_target-mean_sample;
	
	float bloss=0;
	for(unsigned i=0;i!=bias_nepoch;++i)
	{
		auto iter_md=arg_record.begin();
		auto iter_md_batch=arg_record_batch.begin();
		for(unsigned j=0;j!=bias_bsize;++j)
		{
			bias_batch[j]=bias_record[i*bias_bsize+j];
			iter_md_batch=std::copy(iter_md->begin(),iter_md->end(),iter_md_batch);
			++iter_md;
		}
		
		auto iter_mc=arg_random.begin();
		auto iter_mc_batch=arg_random_batch.begin();
		for(unsigned j=0;j!=mc_bsize;++j)
		{
			if(!use_same_bias_factor)
				fes_batch[j]=fes_random[i*mc_bsize+j];
			iter_mc_batch=std::copy(iter_mc->begin(),iter_mc->end(),iter_mc_batch);
			++iter_mc;
		}
		bloss += as_scalar(cg.forward(loss));
		cg.backward(loss);
		trainer_bias->update();
		if(clip_bias)
			nnv.clip_inplace(bias_clip_left,bias_clip_right,clip_bias_last);
		//~ nnv.update_energy_shift();
		//~ nnv.align_zero(bias_zero_args);
	}
	bloss/=bias_nepoch;
	
	return bloss;
}

std::vector<float> Deep_MetaD::hybrid_monte_carlo(const std::vector<float>& init_coords,unsigned cycles)
{
	std::vector<float> coords(init_coords);
	std::vector<float> r_input(coords);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r_input);
	dynet::Expression fes=nnf.energy(cg,inputs);
	
	std::vector<float> deriv;
	std::vector<float> vf0(get_output_and_gradient(cg,inputs,fes,deriv));
	double F0=vf0[0];
	fes_random.push_back(F0);
	arg_random.push_back(coords);
	
	for(unsigned i=1;i!=cycles;++i)
	{
		std::vector<float> r0(coords);
		std::vector<float> dF_dr0(deriv);
		
		std::vector<float> v0(narg);
		double K0=random_velocities(v0);
		double H0=F0+K0;

		double F1=F0;
		for(unsigned j=0;j!=hmc_steps;++j)
		{
			std::vector<float> r1(narg);
			std::vector<float> vh(narg);

			for(unsigned k=0;k!=narg;++k)
			{
				double v=v0[k]-dF_dr0[k]*dt_mc/hmc_arg_mass[k]/2;
				vh[k]=v;
				double r=r0[k]+v*dt_mc;
				r1[k]=r;
				r_input[k]=r;
			}
			
			std::vector<float> dF_dr1;
			std::vector<float> vf1(get_output_and_gradient(cg,inputs,fes,dF_dr1));
			F1=vf1[0];

			std::vector<float> v1(narg);
			for(unsigned k=0;k!=narg;++k)
				v1[k]=vh[k]-dF_dr1[k]*dt_mc/hmc_arg_mass[k]/2;
			
			r0.swap(r1);
			v0.swap(v1);
			dF_dr0.swap(dF_dr1);
		}
		
		double K1=0;
		for(unsigned k=0;k!=narg;++k)
			K1+=hmc_arg_mass[k]*v0[k]*v0[k]/2;
		
		double H1=F1+K1;
		
		bool accept=false;
		if(H1<H0)
			accept=true;
		else
		{
			double acprob=exp(-kBT*(H1-H0));
			double prob=rand_prob(rdgen);
			if(prob<acprob)
				accept=true;
		}
		
		if(accept)
		{
			coords.swap(r0);
			deriv.swap(dF_dr0);
			F0=F1;
		}
		
		fes_random.push_back(F0);
		arg_random.push_back(coords);
	}
	
	return coords;
}

std::vector<float> Deep_MetaD::metropolis_monte_carlo(const std::vector<float>& init_coords,unsigned cycles)
{
	std::vector<float> r0(init_coords);
	std::vector<float> r1(r0);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r1);
	dynet::Expression fes=nnf.energy(cg,inputs);
	
	std::vector<float> out0=dynet::as_vector(cg.forward(fes));
	double E0=out0[0];
	fes_random.push_back(E0);
	arg_random.push_back(r0);
	
	for(unsigned i=1;i!=cycles;++i)
	{	
		random_moves(r1);
		std::vector<float> out1=dynet::as_vector(cg.forward(fes));
		double E1=out1[0];
		
		bool accept=false;
		if(E1<E0)
			accept=true;
		else
		{
			double acprob=exp(-kBT*(E1-E0)/mc_bias_factor);
			double prob=rand_prob(rdgen);
			if(prob<acprob)
				accept=true;
		}
		
		if(accept)
		{
			E0=E1;
			std::copy(r1.begin(),r1.end(),r0.begin());
		}
		else
		{
			E1=E0;
			std::copy(r0.begin(),r0.end(),r1.begin());
		}
		
		fes_random.push_back(E1);
		arg_random.push_back(r1);
	}

	return r1;
}

double Deep_MetaD::random_velocities(std::vector<float>& v)
{
	double kinetics=0;
	for(unsigned i=0;i!=narg;++i)
	{
		double rdv=ndist[i](rdgen);
		v[i]=rdv;
		kinetics+=hmc_arg_mass[i]*rdv*rdv/2;
	}
	return kinetics;
}

void Deep_MetaD::random_moves(std::vector<float>& coords)
{
	for(unsigned i=0;i!=narg;++i)
	{
		double dr=ndist[i](rdgen)*dt_mc;
		coords[i]+=dr;
		if(coords[i]<arg_min[i])
			coords[i]+=arg_period[i];
		else if(coords[i]>arg_max[i])
			coords[i]-=arg_period[i];
	}
}

}
}

#endif
