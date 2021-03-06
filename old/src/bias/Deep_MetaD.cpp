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
	keys.addOutputComponent("rct","default","the c(t) factor to revise the bias potential during updating");
	keys.addOutputComponent("rbias","default","the revised bias potential [V(s)-c(t)]");
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
	keys.add("compulsory","SCALE_FACTOR","20","a constant to scale the output of neural network as bias potential");
	keys.add("compulsory","BIAS_PARAM_OUTPUT","bias_parameters.data","the file to output the parameters of the neural network of the bias potential");
	keys.add("compulsory","FES_PARAM_OUTPUT","fes_parameters.data","the file to output the parameters of the neural network of the free energy surface");
	keys.add("compulsory","BIAS_EPOCH_NUMBER","1","the number of epoch for training the neural networks for bias potential");
	keys.add("compulsory","BATCH_SIZE","250","batch size in each epoch for training neural networks");
	//~ keys.add("compulsory","ZERO_REF_ARG","0","the reference arguments to make the energy of neural networks as zero");
	keys.add("compulsory","LOSS_NORMALIZE_WEIGHT","10","the weight factor of the loss function to normalize the energy of the reference structure to zero");
	keys.add("compulsory","CLIP_LEFT","-0.5","the left value of the range to clip");
	keys.add("compulsory","CLIP_RIGHT","0.5","the right value of the range to clip");
	keys.add("compulsory","MC_SAMPLING_POINTS","1","the sampling points for hybrid monte carlo at a MD simulation step");
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
	
	keys.add("optional","FES_OUTPUT_FILE","the file to output the value of free energy surface");
	keys.add("optional","FES_OUTPUT_STEPS","the frequency to output the file of free energy surface");
	keys.add("optional","BIAS_OUTPUT_FILE","the file to output the bias potential energy");
	keys.add("optional","BIAS_OUTPUT_STEPS","the frequency to output the file of bias potential");
	keys.add("optional","GRID_BINS","the number of bins for the grid to output the energy file");
	keys.add("optional","RANDOM_POINTS_FILE","the file to output the random points of monte carlo");
	keys.add("optional","RANDOM_OUTPUT_STEPS","(default=1) the frequency to output the random points file");
	keys.add("optional","MC_BIAS_FACTOR","the well-tempered bias factor for monte carlo");
	keys.add("optional","LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","HYPER_PARAMS","other hyperparameters for training the neural network");
	keys.add("optional","CLIP_THRESHOLD","the clip threshold for training the neural network");
	keys.add("optional","FES_CLIP_LEFT","the left value of the range to clip the free energy surface");
	keys.add("optional","FES_CLIP_RIGHT","the right value of the range to clip the free energy surface");
	//~ keys.add("optional","FES_ZERO_REF_ARG","the reference arguments to make the free energy as zero");
	keys.add("optional","FES_LOSS_NORMALIZE_WEIGHT","the weight factor of the loss function to normalize the energy of the reference structure to zero");
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
	args_now(getNumberOfArguments(),0),
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
	
	parse("LOSS_NORMALIZE_WEIGHT",bias_loss_normal_weight);
	fes_loss_normal_weight=bias_loss_normal_weight;
	//~ parseVector("ZERO_REF_ARG",bias_zero_args);
	//~ if(bias_zero_args.size()!=narg)
	//~ {
		//~ if(bias_zero_args.size()==1)
		//~ {
			//~ float val=bias_zero_args[0];
			//~ bias_zero_args.assign(narg,val);
		//~ }
		//~ else
			//~ plumed_merror("the size of ZERO_REF_ARG mismatch!");
	//~ }
	//~ fes_zero_args=bias_zero_args;
	
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
	parse("BIAS_OUTPUT_FILE",bias_output_file);
	parse("FES_OUTPUT_FILE",fes_output_file);
	random_output_steps=1;
	bias_output_steps=1;
	fes_output_steps=1;
	parse("RANDOM_OUTPUT_STEPS",random_output_steps);
	parse("BIAS_OUTPUT_STEPS",bias_output_steps);
	parse("FES_OUTPUT_STEPS",fes_output_steps);
	
	parse("SCALE_FACTOR",scale_factor);
	parse("WT_BIAS_FACTOR",bias_factor);
	plumed_massert(bias_factor>0,"WT_BIAS_FACTOR shoud larger than 0");
	mc_bias_factor=bias_factor;
	bias_scale=0;
	
	use_same_bias_factor=true;
	float _mc_bias_factor=-1;
	parse("MC_BIAS_FACTOR",_mc_bias_factor);
	if(_mc_bias_factor>0)
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
	
	mw_size=1;
	dynet_random_seed=0;
	if(use_mw)
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				mw_size=multi_sim_comm.Get_size();
				sw_size=comm.Get_size();
				std::random_device rd;
				dynet_random_seed=rd();
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(dynet_random_seed,0);
			multi_sim_comm.Bcast(sw_size,0);
			multi_sim_comm.Bcast(mw_size,0);
		}
		comm.Barrier();
		comm.Bcast(dynet_random_seed,0);
		comm.Bcast(sw_size,0);
		comm.Bcast(mw_size,0);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			sw_size=comm.Get_size();
			std::random_device rd;
			dynet_random_seed=rd();
		}
		comm.Barrier();
		comm.Bcast(dynet_random_seed,0);
		comm.Bcast(sw_size,0);
	}
	
	ncores=mw_size*sw_size;

	use_mpi=false;
	if(use_mw||sw_size>1)
		use_mpi=true;
	dynet_initialization(dynet_random_seed);
	
	//~ dynet::show_pool_mem_info();
	//~ std::cout<<"Number of multiple walkers: "<<mw_size<<std::endl;
	//~ std::cout<<"Cores of single walkers: "<<sw_size<<std::endl;

	std::string bias_fullname;
	std::string fes_fullname;
	
	parse("MC_SAMPLING_POINTS",mc_points);
	parse("MC_TIMESTEP",dt_mc);
	parseVector("MC_ARG_STDEV",mc_arg_sd);
	
	parseVector("GRID_BINS",grid_bins);
	
	parse("UPDATE_STEPS",update_steps);
	sc_md_points=update_steps;
	sc_mc_points=update_steps*mc_points;
	sw_md_points=sc_md_points;
	sw_mc_points=sc_mc_points*sw_size;
	tot_md_points=sw_md_points*mw_size;
	tot_mc_points=sw_mc_points*mw_size;
	
	parse("BIAS_EPOCH_NUMBER",bias_nepoch);
	parse("BATCH_SIZE",bias_bsize);
	if(update_steps>0)
		plumed_massert(tot_md_points%bias_bsize==0,"UPDATE_STEPS must be divided exactly by BATCH_SIZE");
	fes_bsize=bias_bsize;
	
	parse("CLIP_LEFT",bias_clip_left);
	parse("CLIP_RIGHT",bias_clip_right);
	plumed_massert(bias_clip_right>bias_clip_left,"CLIP_LEFT should be less than CLIP_RIGHT");
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
		mc_start_args.resize(narg*sw_size);
		
		is_random_ouput=false;
		if(random_file.size()>0)
		{
			is_random_ouput=true;
			orandom.link(*this);
			if(use_mw)
			{
				unsigned rank=0;
				if(comm.Get_rank()==0)
					rank=multi_sim_comm.Get_rank();
				comm.Bcast(rank,0);
				if(rank>0)
					random_file="/dev/null";
				orandom.enforceSuffix("");
				//~ std::cout<<rank<<std::endl;
			}
			orandom.open(random_file.c_str());
			orandom.fmtField(" %f");
			orandom.addConstantField("ITERATION");
		}
		
		is_bias_ouput=false;
		if(bias_output_file.size()>0)
		{
			is_bias_ouput=true;
			obias.link(*this);
			if(use_mw)
			{
				unsigned rank=0;
				if(comm.Get_rank()==0)
					rank=multi_sim_comm.Get_rank();
				comm.Bcast(rank,0);
				if(rank>0)
					bias_output_file="/dev/null";
				obias.enforceSuffix("");
			}
			obias.open(bias_output_file.c_str());
			obias.fmtField(" %f");
			obias.addConstantField("ITERATION");
		}
		
		is_fes_ouput=false;
		if(fes_output_file.size()>0)
		{
			is_fes_ouput=true;
			ofes.link(*this);
			if(use_mw)
			{
				unsigned rank=0;
				if(comm.Get_rank()==0)
					rank=multi_sim_comm.Get_rank();
				comm.Bcast(rank,0);
				if(rank>0)
					fes_output_file="/dev/null";
				ofes.enforceSuffix("");
			}
			ofes.open(fes_output_file.c_str());
			ofes.fmtField(" %f");
			ofes.addConstantField("ITERATION");
		}
		
		if(is_bias_ouput||is_fes_ouput)
		{
			if(grid_bins.size()==0)
				grid_bins.assign(narg,100);
			else if(grid_bins.size()!=narg)
			{
				if(grid_bins.size()==1)
				{
					unsigned bin=grid_bins[0];
					grid_bins.assign(narg,bin);
				}
				else
					plumed_merror("the number of GRID_BINS mismatch!");
			}
			grid_space.resize(narg);
			for(unsigned i=0;i!=narg;++i)
			{
				plumed_massert(grid_bins[i]>0,"the value of GRID_BINS must be larger than 0!");
				grid_space[i]=(arg_max[i]-arg_min[i])/grid_bins[i];
				if(!arg_pbc[i])
					++grid_bins[i];
			}
		}
		
		if(use_diff_param)
		{
			parse("FES_OPT_ALGORITHM",fes_algorithm);
			parse("FES_LOSS_NORMALIZE_WEIGHT",fes_loss_normal_weight);
			
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
			
			//~ std::vector<float> _fes_zero_args;
			//~ parseVector("FES_ZERO_REF_ARG",_fes_zero_args);
			//~ if(_fes_zero_args.size()>0)
				//~ fes_zero_args=_fes_zero_args;
			//~ if(fes_zero_args.size()!=narg)
			//~ {
				//~ if(fes_zero_args.size()==1)
				//~ {
					//~ float val=fes_zero_args[0];
					//~ fes_zero_args.assign(narg,val);
				//~ }
				//~ else
					//~ plumed_merror("the size of FES_ZERO_REF_ARG mismatch!");
			//~ }
			
			parse("FES_CLIP_LEFT",fes_clip_left);
			parse("FES_CLIP_RIGHT",fes_clip_right);
			plumed_massert(fes_clip_right>fes_clip_left,"FES_CLIP_LEFT should be less than FES_CLIP_RIGHT");
		}
		bias_nbatch=tot_md_points/bias_bsize;
		fes_nbatch=tot_mc_points/fes_bsize;
		plumed_massert(tot_mc_points%(bias_nepoch*bias_nbatch)==0,"the number of MC samping points in each walker must be divided exactly by BATCH_SIZE*BIAS_EPOCH_NUMBER");
		mc_bsize=tot_mc_points/(bias_nepoch*bias_nbatch);
		log_fes_bsize=std::log(static_cast<double>(fes_bsize));
		
		//~ nnv.set_zero_cvs(bias_zero_args);
		//~ nnf.set_zero_cvs(fes_zero_args);
		
		md_bsize=fes_bsize;
		if(fes_bsize>tot_md_points)
			md_bsize=tot_md_points;
		md_nbatch=tot_md_points/md_bsize;
		
		for(unsigned i=0;i!=tot_md_points;++i)
			md_ids.push_back(i);
		for(unsigned i=0;i!=tot_mc_points;++i)
			mc_ids.push_back(i);
		
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
		addComponent("rct"); componentIsNotPeriodic("rct");
		value_rct=getPntrToComponent("rct");
		addComponent("rbias"); componentIsNotPeriodic("rbias");
		value_rbias=getPntrToComponent("rbias");
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
	log.printf("  with random seed for DyNet: %d\n",int(dynet_random_seed));
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
		
		log.printf("  with sampling of CVs from MD simulation: \n");
		log.printf("    with sampling cycle for each traning update: %d.\n",int(update_steps));
		log.printf("    with sampling points of a update cycle for each CPU core: %d.\n",int(sc_md_points));
		log.printf("    with sampling points of a update cycle for each walker: %d.\n",int(sw_md_points));
		log.printf("    with total sampling points for each update cycle: %d.\n",int(tot_md_points));
		
		if(use_hmc)
			log.printf("  using hybrid monte carlo for the sampling of free energy surface.\n");
		else
			log.printf("  using metropolis monte carlo for the sampling of free energy surface.\n");
		log.printf("    with well-tempered bias factor: %f.\n",mc_bias_factor);
		log.printf("    with sampling points per MD step: %d.\n",int(mc_points));
		log.printf("    with sampling points of a update cycle for each CPU core: %d.\n",int(sc_mc_points));
		log.printf("    with sampling points of a update cycle for each walker: %d.\n",int(sw_mc_points));
		log.printf("    with total sampling points for each update cycle: %d.\n",int(tot_mc_points));
		log.printf("    with time step: %f.\n",dt_mc);
		if(use_hmc)
		{
			log.printf("    with moving steps for each sampling point: %d.\n",int(hmc_steps));
			for(unsigned i=0;i!=narg;++i)
				log.printf("    %dth CV \"%s\" with standard deviation %f and mass %f.\n",int(i+1),arg_label[i].c_str(),mc_arg_sd[i],hmc_arg_mass[i]);
		}
		else
		{
			for(unsigned i=0;i!=narg;++i)
				log.printf("    %dth CV \"%s\" with standard deviation %f.\n",int(i+1),arg_label[i].c_str(),mc_arg_sd[i]);
		}
		
		log.printf("  update the parameters of neural networks every %d steps.\n",int(update_steps));
		log.printf("  using optimization algorithm for the neural networks of bias potential (political network): %s.\n",bias_fullname.c_str());
		log.printf("    with batch size for traning bias potential at political networks: %d.\n",int(bias_bsize));
		log.printf("    with batch size for traning free energy surface at political networks: %d.\n",int(mc_bsize));
		log.printf("    with number of epoch of mini batches for the political networks: %d.\n",int(bias_nbatch));
		log.printf("    with weight for the loss fucntion to normalize the bias potential energy: %f.\n",bias_loss_normal_weight);
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
		log.printf("    with number of epoch of mini batches for the value networks: %d.\n",int(fes_nbatch));
		log.printf("    with weight for the loss fucntion to normalize the free energy surface: %f.\n",fes_loss_normal_weight);
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
		if(is_random_ouput)
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
			nnf.set_hidden_layers(ldf,aff);
			nnf.build_neural_network(pcf);
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
	for(unsigned i=0;i!=narg;++i)
	{
		float val=getArgument(i);
		args_now[i]=val;
	}

	std::vector<float> deriv;
	bias_now=calc_bias(args_now,deriv);
	
	setBias(bias_now);
	value_rbias->set(bias_now-rct);
	
	for(unsigned i=0;i!=narg;++i)
	{
		double bias_force=-deriv[i];
		setOutputForce(i,bias_force);
	}
	
	++steps;
}

void Deep_MetaD::update()
{
	if(update_steps>0)
	{
		std::vector<float> deriv;
		double fes=calc_fes(args_now,deriv);
		value_fes->set(fes);
		
		bias_record.push_back(bias_now);
		std::copy(args_now.begin(),args_now.end(),std::back_inserter(arg_record));

		if(steps==1)
		{
			for(unsigned i=0;i!=sw_size;++i)
			{
				for(unsigned j=0;j!=narg;++j)
					mc_start_args[i*narg+j]=args_now[j];
			}
		}
		if(steps%update_steps==1)
		{
			for(unsigned j=0;j!=narg;++j)
				mc_start_args[j]=args_now[j];
		}
		
		if(use_hmc)
			hybrid_monte_carlo(mc_points);
		else
			metropolis_monte_carlo(mc_points);

		if(steps%update_steps==0)
		{
			std::vector<float> all_bias_record(tot_md_points,0);
			std::vector<float> all_fes_random(tot_mc_points,0);
			std::vector<float> all_arg_record(tot_md_points*narg,0);
			std::vector<float> all_arg_random(tot_mc_points*narg,0);
			if(use_mw)
			{
				if(comm.Get_rank()==0)
				{				
					multi_sim_comm.Allgather(bias_record,all_bias_record);
					multi_sim_comm.Allgather(fes_random,all_fes_random);
					multi_sim_comm.Allgather(arg_record,all_arg_record);
					multi_sim_comm.Allgather(arg_random,all_arg_random);
				}
				comm.Barrier();
				comm.Bcast(all_bias_record,0);
				comm.Bcast(all_fes_random,0);
				comm.Bcast(all_arg_record,0);
				comm.Bcast(all_arg_random,0);
			}
			
			std::vector<float> new_arg_record;
			std::vector<float> new_bias_record;
			std::vector<float> new_arg_random;
			std::vector<float> new_fes_random;
			std::vector<float> update_fes_random;
			
			std::vector<float> param_fes;
			std::vector<float> param_bias;
			
			double floss=0,bloss=0;
			if(use_mw)
			{
				param_fes.resize(nnf.parameters_number());
				param_bias.resize(nnv.parameters_number());
				
				if(comm.Get_rank()==0)
				{
					if(multi_sim_comm.Get_rank()==0)
					{
						floss=update_fes(all_arg_record,all_bias_record,
							all_arg_random,all_fes_random,
							new_arg_record,new_bias_record,
							new_arg_random,new_fes_random,
							update_fes_random);
						bloss=update_bias(new_arg_record,new_bias_record,
							new_arg_random,new_fes_random,
							update_fes_random);
						
						new_arg_record.resize(0);
						new_bias_record.resize(0);
						new_arg_random.resize(0);
						new_fes_random.resize(0);
						update_fes_random.resize(0);
						
						param_fes=nnf.get_parameters();
						param_bias=nnv.get_parameters();
					}
					multi_sim_comm.Barrier();
					multi_sim_comm.Bcast(rct,0);
					multi_sim_comm.Bcast(floss,0);
					multi_sim_comm.Bcast(bloss,0);
					multi_sim_comm.Bcast(param_fes,0);
					multi_sim_comm.Bcast(param_bias,0);
				}
				comm.Barrier();
				comm.Bcast(rct,0);
				comm.Bcast(floss,0);
				comm.Bcast(bloss,0);
				comm.Bcast(param_fes,0);
				comm.Bcast(param_bias,0);
				nnf.set_parameters(param_fes);
				nnv.set_parameters(param_bias);
			}
			else
			{
				if(sw_size>1)
				{
					param_fes.resize(nnf.parameters_number());
					param_bias.resize(nnv.parameters_number());
				}
				if(comm.Get_rank()==0)
				{
					floss=update_fes(arg_record,bias_record,arg_random,fes_random,
						new_arg_record,new_bias_record,new_arg_random,new_fes_random,
						update_fes_random);
					bloss=update_bias(new_arg_record,new_bias_record,
						new_arg_random,new_fes_random,update_fes_random);
					
					new_arg_record.resize(0);
					new_bias_record.resize(0);
					new_arg_random.resize(0);
					new_fes_random.resize(0);
					update_fes_random.resize(0);
					
					if(sw_size>1)
					{
						param_fes=nnf.get_parameters();
						param_bias=nnv.get_parameters();
					}
				}
				if(sw_size>1)
				{
					comm.Barrier();
					comm.Bcast(rct,0);
					comm.Bcast(floss,0);
					comm.Bcast(bloss,0);
					comm.Bcast(param_fes,0);
					comm.Bcast(param_bias,0);
					
					nnf.set_parameters(param_fes);
					nnv.set_parameters(param_bias);
				}
			}
			
			if(is_random_ouput&&update_cycle%random_output_steps==0)
			{
				if(!use_mw)
				{
					all_arg_random.swap(arg_random);
					all_fes_random.swap(fes_random);
				}
				orandom.printField("ITERATION",getTime());
				for(unsigned i=0;i!=tot_mc_points;++i)
				{
					orandom.printField("index",int(i));
					for(unsigned j=0;j!=narg;++j)
						orandom.printField(arg_label[j],all_arg_random[i*narg+j]);
					orandom.printField("fes",all_fes_random[i]);
					orandom.printField();
				}
				orandom.flush();
			}
			
			if(is_bias_ouput&&update_cycle%bias_output_steps==0)
				write_bias_file();
			if(is_fes_ouput&&update_cycle%fes_output_steps==0)
				write_fes_file();
						
			value_rct->set(rct);
			value_floss->set(floss);
			value_bloss->set(bloss);
			bias_record.resize(0);
			arg_record.resize(0);
			fes_random.resize(0);
			arg_random.resize(0);

			++update_cycle;
		}
	}
}

float Deep_MetaD::update_fes(const std::vector<float>& old_arg_record,
	const std::vector<float>& old_bias_record,
	const std::vector<float>& old_arg_random,
	const std::vector<float>& old_fes_random,
	std::vector<float>& new_arg_record,
	std::vector<float>& new_bias_record,
	std::vector<float>& new_arg_random,
	std::vector<float>& new_fes_random,
	std::vector<float>& update_fes_random)
{
	std::vector<float> origin_fes_batch(fes_bsize);
	std::vector<float> update_fes_batch(fes_bsize);
	std::vector<float> arg_random_batch(narg*fes_bsize);
	std::vector<float> bias_batch(md_bsize);
	std::vector<float> arg_record_batch(narg*md_bsize);

	dynet::ComputationGraph cg;

	dynet::Dim bias_dim({md_bsize});
	dynet::Dim fes_dim({fes_bsize});
	dynet::Dim md_dim({narg,md_bsize});
	dynet::Dim mc_dim({narg,fes_bsize});

	dynet::Expression bias_inputs=dynet::input(cg,bias_dim,&bias_batch);
	dynet::Expression fes_old=dynet::input(cg,fes_dim,&origin_fes_batch);
	dynet::Expression fes_new=dynet::input(cg,fes_dim,&update_fes_batch);
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record_batch);
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random_batch);

	dynet::Expression nnf_md=nnf.MLP_output(cg,md_inputs);
	dynet::Expression nnf_mc=nnf.MLP_output(cg,mc_inputs);
	
	dynet::Expression fes_mc=energy_scale*nnf_mc;
	dynet::Expression weight_bias_mc=fes_old/mc_bias_factor-fes_new-dynet::transpose(fes_mc);
	dynet::Expression log_Z_mc=dynet::logsumexp_dim(beta*weight_bias_mc,0);
	
	dynet::Expression md_weights=dynet::softmax(beta*bias_inputs);
	dynet::Expression mc_weights=dynet::softmax(beta*(fes_old/mc_bias_factor-fes_new));
	
	dynet::Expression mean_sample=nnf_md*md_weights;
	dynet::Expression mean_target=nnf_mc*mc_weights;
	
	dynet::Expression loss_Z=0.5*dynet::squared_norm(log_Z_mc-log_fes_bsize);
	dynet::Expression loss=mean_sample-mean_target+loss_Z*fes_loss_normal_weight;

	//~ unsigned seed=std::time(0);
	//~ std::shuffle(fes_random.begin(),fes_random.end(),std::default_random_engine(seed));
	//~ std::shuffle(arg_random.begin(),arg_random.end(),std::default_random_engine(seed));
	std::random_shuffle(mc_ids.begin(),mc_ids.end());

	unsigned imc=0;
	unsigned imd=0;

	double floss=0;
	
	for(unsigned i=0;i!=fes_nbatch;++i)
	{
		unsigned iarg=0;
		for(unsigned j=0;j!=fes_bsize;++j)
		{
			unsigned mcid=mc_ids[imc++];	
			float fes=old_fes_random[mcid];
			origin_fes_batch[j]=fes;
			
			new_fes_random.push_back(fes);
			for(unsigned k=0;k!=narg;++k)
			{
				float arg=old_arg_random[mcid*narg+k];
				arg_random_batch[iarg++]=arg;
				new_arg_random.push_back(arg);
			}
		}
		
		if(i%md_nbatch==0)
		{
			if(i==0||md_nbatch>1)
				std::random_shuffle(md_ids.begin(),md_ids.end());
			if(md_nbatch>1)
				imd=0;
		}
		if(i==0||md_nbatch>1)
		{
			iarg=0;
			for(unsigned j=0;j!=md_bsize;++j)
			{
				unsigned mdid=md_ids[imd++];
				float vbias=old_bias_record[mdid];
				bias_batch[j]=vbias;
				
				if(i<md_nbatch)
					new_bias_record.push_back(vbias);
				for(unsigned k=0;k!=narg;++k)
				{
					float arg=old_arg_record[mdid*narg+k];
					arg_record_batch[iarg++]=arg;
					if(i<md_nbatch)
						new_arg_record.push_back(arg);
				}
			}
		}
		
		if(i==0)
			std::copy(origin_fes_batch.begin(),origin_fes_batch.end(),update_fes_batch.begin());
		else
			update_fes_batch=dynet::as_vector(cg.forward(fes_mc));
		
		floss += as_scalar(cg.forward(loss));
		cg.backward(loss);
		trainer_fes->update();
		if(clip_fes)
			nnf.clip_inplace(fes_clip_left,fes_clip_right,clip_fes_last);
	}
	floss/=fes_nbatch;
	
	dynet::Dim arg_dim({narg,tot_mc_points});
	dynet::Dim e_dim({tot_mc_points});
	dynet::Expression tot_arg_random=dynet::input(cg,arg_dim,&new_arg_random);
	dynet::Expression tot_fes_old=dynet::input(cg,e_dim,&new_fes_random);
	dynet::Expression tot_fes_new=nnf.energy(cg,tot_arg_random);
	
	update_fes_random=dynet::as_vector(cg.forward(tot_fes_new));
	
	dynet::Expression m_beta_F=beta*(tot_fes_old/mc_bias_factor-2*dynet::transpose(tot_fes_new));
	dynet::Expression tot_log_Z=dynet::logsumexp_dim(m_beta_F,0);
	std::vector<float> out=dynet::as_vector(cg.forward(tot_log_Z));
	log_Z=out[0];
	
	return floss;
}

float Deep_MetaD::update_bias(const std::vector<float>& vec_arg_record,
	const std::vector<float>& vec_bias_record,
	const std::vector<float>& vec_arg_random,
	const std::vector<float>& origin_fes_random,
	const std::vector<float>& update_fes_random)
{
	dynet::ComputationGraph cg;
	
	std::vector<float> arg_record_batch(narg*bias_bsize);
	std::vector<float> arg_random_batch(narg*mc_bsize);
	std::vector<float> origin_fes_batch(mc_bsize);
	std::vector<float> update_fes_batch(mc_bsize);
	std::vector<float> origin_bias_batch(bias_bsize);
	std::vector<float> update_bias_batch(bias_bsize);

	dynet::Dim md_dim({narg,bias_bsize});
	dynet::Dim fes_dim({mc_bsize});
	dynet::Dim mc_dim({narg,mc_bsize});
	
	dynet::Expression fes_old=dynet::input(cg,fes_dim,&origin_fes_batch);
	dynet::Expression fes_new=dynet::input(cg,fes_dim,&update_fes_batch);
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record_batch);
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random_batch);
	
	dynet::Expression nnv_mc=nnv.MLP_output(cg,mc_inputs);
	dynet::Expression bias_mc=dynet::transpose(energy_scale*nnv_mc);
	dynet::Expression nnv_md=nnv.MLP_output(cg,md_inputs);
	
	dynet::Expression log_wbias_mc=fes_old/mc_bias_factor-fes_new/bias_factor;
	
	//~ dynet::Expression log_m_F=bias_mc*bias_factor/(bias_factor-1.0)+fes_old/mc_bias_factor-fes_new;
	//~ dynet::Expression log_Z_mc=dynet::logsumexp_dim(beta*log_m_F,0);
	//~ dynet::Expression log_m_F_p_V=bias_mc/(bias_factor-1.0)+log_wbias_mc;
	//~ dynet::Expression log_Zb_mc=dynet::logsumexp_dim(beta*log_m_F_p_V,0);
	float rct_scale=(bias_factor+1.0)/bias_factor;
	dynet::Expression log_m_F_p_V=fes_old/mc_bias_factor-fes_new*rct_scale-bias_mc;
	dynet::Expression log_Zb=dynet::logsumexp_dim(beta*log_m_F_p_V,0);
	
	//~ dynet::Expression dy_rct=log_Z_mc-log_Zb_mc;
	dynet::Expression dy_rct=log_Z-log_Zb;
	
	dynet::Expression loss_rct=0.5*dynet::squared_norm(dy_rct);
	
	dynet::Expression mc_weights=dynet::softmax(beta*log_wbias_mc);
	dynet::Expression mean_target=nnv_mc*mc_weights;
	
	dynet::Expression mean_sample;
	
	dynet::Expression bias_md=energy_scale*nnv_md;
	if(bias_nbatch==1)
	{
		arg_record_batch=vec_arg_record;
		mean_sample=dynet::mean_elems(nnv_md);
	}
	else
	{
		dynet::Dim bias_dim({bias_bsize});
		dynet::Expression bias_old=dynet::input(cg,bias_dim,&origin_bias_batch);
		dynet::Expression bias_new=dynet::input(cg,bias_dim,&update_bias_batch);
		dynet::Expression md_weights=dynet::softmax(beta*(bias_old-bias_new));
		
		mean_sample=nnv_md*md_weights;
	}
	
	dynet::Expression loss=mean_target-mean_sample+loss_rct*bias_loss_normal_weight;
	
	if(bias_nbatch>1)
	{
		for(unsigned i=0;i!=md_ids.size();++i)
			md_ids[i]=i;
	}
	
	float bloss=0;
	for(unsigned iepoch=0;iepoch!=bias_nepoch;++iepoch)
	{
		if(iepoch>1&&bias_nbatch>1)
			std::random_shuffle(mc_ids.begin(),mc_ids.end());
		
		unsigned iarg_record=0;
		unsigned iarg_random=0;
		for(unsigned i=0;i!=bias_nbatch;++i)
		{	
			unsigned iarg_batch=0;
			for(unsigned j=0;j!=mc_bsize;++j)
			{
				origin_fes_batch[j]=origin_fes_random[i*mc_bsize+j];
				update_fes_batch[j]=update_fes_random[i*mc_bsize+j];
				for(unsigned k=0;k!=narg;++k)
					arg_random_batch[iarg_batch++]=vec_arg_random[iarg_random++];
			}
			
			if(bias_nbatch>1)
			{
				iarg_batch=0;
				for(unsigned j=0;j!=bias_bsize;++j)
				{
					unsigned mdid=md_ids[iarg_record++];
					origin_bias_batch[j]=vec_bias_record[mdid];
					for(unsigned k=0;k!=narg;++k)
						arg_record_batch[iarg_batch++]=vec_arg_record[mdid*narg+k];
				}
				if(i==0)
					std::copy(origin_bias_batch.begin(),origin_bias_batch.end(),update_bias_batch.begin());
				else
					update_bias_batch=dynet::as_vector(cg.forward(bias_md));
			}

			bloss += as_scalar(cg.forward(loss));
			cg.backward(loss);
			
			trainer_bias->update();
			if(clip_bias)
				nnv.clip_inplace(bias_clip_left,bias_clip_right,clip_bias_last);
		}
	}
	
	bloss/=bias_nbatch*bias_nepoch;
	
	dynet::Dim arg_dim({narg,tot_mc_points});
	dynet::Dim e_dim({tot_mc_points});
	dynet::Expression arg_input=dynet::input(cg,arg_dim,&vec_arg_random);
	dynet::Expression tot_fes_old=dynet::input(cg,e_dim,&origin_fes_random);
	dynet::Expression tot_fes_new=dynet::input(cg,e_dim,&update_fes_random);
	dynet::Expression tot_bias_new=dynet::transpose(nnv.energy(cg,arg_input));
	dynet::Expression fin_log_m_F_p_V=tot_fes_old/mc_bias_factor-tot_fes_new*(bias_factor+1.0)/bias_factor-tot_bias_new;
	dynet::Expression fin_log_Zb=dynet::logsumexp_dim(beta*fin_log_m_F_p_V,0);
	
	std::vector<float> out=dynet::as_vector(cg.forward(fin_log_Zb));
	rct=(log_Z-out[0])/beta;
	
	return bloss;
}

void Deep_MetaD::hybrid_monte_carlo(unsigned cycles)
{
	unsigned stride;
	std::vector<unsigned> mc_seeds(sw_size);
	if(comm.Get_rank()==0)
	{
		stride=comm.Get_size();
		std::random_device rd;
		for(unsigned i=0;i!=sw_size;++i)
			mc_seeds[i]=rd();
	}
	comm.Barrier();
	comm.Bcast(stride,0);
	comm.Bcast(mc_seeds,0);
	
	std::vector<float> init_coords(narg);
	const unsigned rank=comm.Get_rank();
	rdgen.seed(mc_seeds[rank]);
	for(unsigned i=0;i!=narg;++i)
		init_coords[i]=mc_start_args[rank*narg+i];
	
	std::vector<float> coords(init_coords);
	std::vector<float> r_input(coords);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r_input);
	dynet::Expression fes=nnf.energy(cg,inputs);
	
	std::vector<float> deriv;
	std::vector<float> vf0(get_output_and_gradient(cg,inputs,fes,deriv));
	double F0=vf0[0];
	
	unsigned mc_cycles=cycles*sw_size;
	std::vector<float> fes_gen(mc_cycles,0);
	std::vector<float> arg_gen(narg*mc_cycles,0);
	
	mc_start_args.assign(narg*sw_size,0);
	for(unsigned i=rank;i<mc_cycles;i+=stride)
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
			double acprob=exp(-beta*(H1-H0)/mc_bias_factor);
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
		
		fes_gen[i]=F0;
		for(unsigned j=0;j!=narg;++j)
		{
			arg_gen[i*narg+j]=coords[j];
			mc_start_args[narg*rank+j]=coords[j];
		}
	}
	comm.Sum(mc_start_args);
	comm.Sum(fes_gen);
	comm.Sum(arg_gen);
	
	std::copy(fes_gen.begin(),fes_gen.end(),std::back_inserter(fes_random));
	std::copy(arg_gen.begin(),arg_gen.end(),std::back_inserter(arg_random));
}

void Deep_MetaD::metropolis_monte_carlo(unsigned cycles)
{
	unsigned stride;
	std::vector<unsigned> mc_seeds(sw_size);
	if(comm.Get_rank()==0)
	{
		stride=comm.Get_size();
		std::random_device rd;
		for(unsigned i=0;i!=sw_size;++i)
			mc_seeds[i]=rd();
	}
	comm.Barrier();
	comm.Bcast(stride,0);
	comm.Bcast(mc_seeds,0);
	
	std::vector<float> init_coords(narg);
	const unsigned rank=comm.Get_rank();
	rdgen.seed(mc_seeds[rank]);
	for(unsigned i=0;i!=narg;++i)
		init_coords[i]=mc_start_args[rank*narg+i];
	
	std::vector<float> r0(init_coords);
	std::vector<float> r1(r0);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r1);
	dynet::Expression fes=nnf.energy(cg,inputs);
	
	std::vector<float> out0=dynet::as_vector(cg.forward(fes));
	double E0=out0[0];
	
	unsigned mc_cycles=cycles*sw_size;
	std::vector<float> fes_gen(mc_cycles,0);
	std::vector<float> arg_gen(narg*mc_cycles,0);
	
	mc_start_args.assign(narg*sw_size,0);
	for(unsigned i=rank;i<mc_cycles;i+=stride)
	{
		random_moves(r1);
		std::vector<float> out1=dynet::as_vector(cg.forward(fes));
		double E1=out1[0];
		
		bool accept=false;
		if(E1<E0)
			accept=true;
		else
		{
			double acprob=exp(-beta*(E1-E0)/mc_bias_factor);
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
		
		fes_gen[i]=E1;
		for(unsigned j=0;j!=narg;++j)
		{
			arg_gen[i*narg+j]=r0[j];
			mc_start_args[narg*rank+j]=r0[j];
		}
	}
	comm.Sum(mc_start_args);
	comm.Sum(fes_gen);
	comm.Sum(arg_gen);
	
	std::copy(fes_gen.begin(),fes_gen.end(),std::back_inserter(fes_random));
	std::copy(arg_gen.begin(),arg_gen.end(),std::back_inserter(arg_random));
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

void Deep_MetaD::write_grid_file(MLP_energy& nn,OFile& ogrid,const std::string& label)
{	
	unsigned r=0;
	if(use_mw)
	{
		if(comm.Get_rank()==0)
			r=multi_sim_comm.Get_rank();
		comm.Bcast(r,0);
	}
	if(r==0)
		ogrid.rewind();
		
	dynet::ComputationGraph cg;
	
	std::vector<float> args(narg);
	dynet::Expression arg_inputs=dynet::input(cg,{narg},&args);
	dynet::Expression output=nn.energy(cg,arg_inputs);
	
	ogrid.addConstantField("ITERATION");
	ogrid.printField("ITERATION",getTime());
	
	std::vector<unsigned> id(narg,0);
	
	unsigned count=0;
	bool do_cycle=true;
	while(do_cycle)
	{
		ogrid.printField("Index",int(count++));
		
		for(unsigned i=0;i!=narg;++i)
		{
			float val=arg_min[i]+id[i]*grid_space[i];
			ogrid.printField(arg_label[i],val);
			args[i]=val;
		}
		std::vector<float> out=dynet::as_vector(cg.forward(output));
		ogrid.printField(label,out[0]);
		ogrid.printField();
		
		++id[0];
		for(unsigned i=0;i!=narg;++i)
		{
			if(id[i]<grid_bins[i])
				break;
			else
			{
				if((i+1)==narg)
					do_cycle=false;
				else
				{
					if((i+2)==narg)
						ogrid.printField();
					++id[i+1];
					id[i]=0;
				}
			}
		}
	}
	ogrid.flush();
}


}
}

#endif
