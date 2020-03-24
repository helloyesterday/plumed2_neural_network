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

#include "NNB_Deep_MetaD.h"
#include "ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/Exception.h"
#include "tools/Matrix.h"
#include "tools/Random.h"
#include "tools/Tools.h"
#include "tools/DynetTools.h"
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

PLUMED_REGISTER_ACTION(NNB_Deep_MetaD,"NNB_DEEP_METAD")

void NNB_Deep_MetaD::registerKeywords(Keywords& keys)
{
	NN_Bias::registerKeywords(keys);
	ActionWithValue::useCustomisableComponents(keys);
	keys.addOutputComponent("rct","default","the c(t) factor to revise the bias potential during updating");
	keys.addOutputComponent("rbias","default","the revised bias potential [V(s)-c(t)]");
	keys.addOutputComponent("fes","default","loss function for the bias potential (political network)");
	keys.addOutputComponent("Vloss","default","loss function for the bias potential (political network)");
	keys.addOutputComponent("Floss","default","loss function of the free energy surface (value network)");
	keys.remove("ARG");
    keys.add("optional","ARG","the argument(CVs) here must be the potential energy. If no argument is setup, only bias will be integrated in this method."); 
	//~ keys.use("ARG");

	keys.add("compulsory","FES_NNCV","the neural network collective variable for free energy surface");
	keys.add("compulsory","ALGORITHM_BIAS","the optimization algorithm to train the neural networks of bias potential");
	keys.add("compulsory","ALGORITHM_FES","the optimization algorithm to train the neural networks of free energy surface");
	keys.add("compulsory","UPDATE_STEPS","1000","the frequency to update parameters of neural networks. ZERO means do not update the parameters");
	keys.add("compulsory","BIAS_PARAM_OUTPUT","bias_parameters.data","the file to output the parameters of the neural network of the bias potential");
	keys.add("compulsory","FES_PARAM_OUTPUT","fes_parameters.data","the file to output the parameters of the neural network of the free energy surface");
	keys.add("compulsory","BIAS_EPOCH_NUMBER","1","the number of epoch for training the neural networks for bias potential");
	keys.add("compulsory","BATCH_SIZE","250","batch size in each epoch for training neural networks");
	keys.add("compulsory","BIAS_LOSS_WEIGHT","10","the weight factor of the loss function to normalize the energy of the reference structure to zero");
	keys.add("compulsory","FES_LOSS_WEIGHT","10","the weight factor of the loss function to normalize the energy of the reference structure to zero");
	keys.add("compulsory","MC_SAMPLING_POINTS","1","the sampling points for hybrid monte carlo at a MD simulation step");
	keys.add("compulsory","MC_TIMESTEP","0.001","the time step for hybrid monte carlo for the sampling of the neural network of free energy");
	keys.add("compulsory","MC_ARG_STDEV","1.0","the standard deviation of the normal distribution to generate random velocity for each argument at hybrid monte carlo");
	keys.add("compulsory","HMC_STEPS","10","the steps for a sampling of a hybrid monte carlo run");
	keys.add("compulsory","HMC_ARG_MASS","1.0","the mass for each argument at hybrid monte carlo");
	keys.add("compulsory","WT_BIAS_FACTOR","10","the well-tempered bias factor for deep metadynamics");
	
	keys.addFlag("USE_HYBRID_MONTE_CARLO",false,"use hybrid monte carlo to sample the neural networks");
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
	
	keys.add("optional","FES_SCALE","use a different factor to scale the output of neural network as free energy surface potential. the default value is equal to the scale factor for bias potential."); 
	keys.add("optional","RANDOM_POINTS_FILE","the file to output the random points of monte carlo");
	keys.add("optional","RANDOM_OUTPUT_STEPS","(default=1) the frequency to output the random points file");
	keys.add("optional","MC_BIAS_FACTOR","the well-tempered bias factor for monte carlo");
	keys.add("optional","FES_BATCH_SIZE","batch size in each epoch for training free energy surface");
}

NNB_Deep_MetaD::NNB_Deep_MetaD(const ActionOptions& ao):
	Action(ao),
	NN_Bias(ao),
	firsttime(true),
	use_same_gamma(true),
	step(0),
	update_cycle(0),
	rand_prob(0,1.0),
	nncv_bias(get_nncv_ptr()),
	nncv_fes(NULL),
	algorithm_bias(NULL),
	algorithm_fes(NULL),
	trainer_bias(NULL),
	trainer_fes(NULL),
	value_rct(NULL),
	value_rbias(NULL),
	value_fes(NULL),
	value_Vloss(NULL),
	value_Floss(NULL)
{
	parse("UPDATE_STEPS",update_steps);
	parseFlag("MULTIPLE_WALKERS",use_mw);
	mw_size=1;
	sw_rank=comm.Get_rank();
	if(use_mw)
	{
		if(comm.Get_rank()==0)
		{
			mw_rank=multi_sim_comm.Get_rank();
			if(multi_sim_comm.Get_rank()==0)
			{
				mw_size=multi_sim_comm.Get_size();
				sw_size=comm.Get_size();
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(sw_size,0);
			multi_sim_comm.Bcast(mw_size,0);
		}
		comm.Barrier();
		comm.Bcast(mw_rank,0);
		comm.Bcast(sw_size,0);
		comm.Bcast(mw_size,0);
	}
	else
	{
		mw_rank=0;
		if(comm.Get_rank()==0)
		{
			sw_size=comm.Get_size();
		}
		comm.Barrier();
		comm.Bcast(sw_size,0);
	}
	ncores=mw_size*sw_size;
	
	sc_md_points=update_steps;
	sc_mc_points=update_steps*mc_points;
	sw_md_points=sc_md_points;
	sw_mc_points=sc_mc_points*sw_size;
	tot_md_points=sw_md_points*mw_size;
	tot_mc_points=sw_mc_points*mw_size;
	
	parse("FES_NNCV",fes_nncv_label);
	parse("ALGORITHM_BIAS",opt_bias_label);
	parse("ALGORITHM_FES",opt_fes_label);
	
	if(update_steps>0)
	{
		nncv_fes=plumed.getActionSet().selectWithLabel<NNCV*>(fes_nncv_label);
		if(!nncv_fes)
			plumed_merror("Neural network collective variable \""+fes_nncv_label+"\" does not exist. FES_NNCV must be NNCV type and NN_BIAS should always be defined AFTER NNCV.");
		nncv_fes->linkBias(this);
		plumed_massert(nncv_fes->get_input_number()==nncv_bias->get_input_number(),"The input dimension of FES_NNCV \""+fes_nncv_label+"\" must be equal to the input of bias potential \""+get_nncv_label()+"\".");
		plumed_massert(nncv_fes->get_output_dim()==1,"The output dimension of FES_NNCV \""+fes_nncv_label+"\" must be ONE when it used as the input of a bias potential.");
		narg=nncv_bias->get_input_number();
		
			algorithm_bias=plumed.getActionSet().selectWithLabel<Trainer_Algorithm*>(opt_bias_label);
		if(!algorithm_bias)
			plumed_merror("OPT_ALGORITHM \""+opt_bias_label+"\" does not exist. ALGORITHM_BIAS must be OPT_ALGORITHM type and DEEP_METAD should always be defined AFTER OPT_ALGORITHM.");
		algorithm_fes=plumed.getActionSet().selectWithLabel<Trainer_Algorithm*>(opt_fes_label);
		if(!algorithm_fes)
			plumed_merror("OPT_ALGORITHM \""+opt_fes_label+"\" does not exist. ALGORITHM_FES must be OPT_ALGORITHM type and DEEP_METAD should always be defined AFTER OPT_ALGORITHM.");
			
		trainer_bias=algorithm_bias->new_trainer(nnv->get_model());
		trainer_fes=algorithm_fes->new_trainer(nnf->get_model());
	}
	
	bias_scale=energy_scale;
	fes_sacle=energy_scale;
	if(keywords.exists("FES_SCALE"))
	{
		float fes_factor;
		parse("FES_SCALE",fes_factor);
		fes_scale=fes_factor*kBT;
	}
	
	if(nncv_bias->get_grid_wstride()>0)
		nncv_bias->set_grid_output_scale(bias_scale);
	if(nncv_fes->get_grid_wstride()>0)
		nncv_fes->set_grid_output_scale(fes_scale);
	
	parse("WT_BIAS_FACTOR",wt_gamma);
	plumed_massert(wt_gamma>0,"WT_BIAS_FACTOR shoud larger than 0");
	mc_gamma=wt_gamma;
	
	float mc_bias_factor=-1;
	parse("MC_BIAS_FACTOR",mc_bias_factor);
	if(mc_bias_factor>0)
	{
		use_same_gamma=false;
		mc_gamma=mc_bias_factor;
	}
	
	parse("MC_SAMPLING_POINTS",mc_points);
	parse("MC_TIMESTEP",dt_mc);
	parseVector("MC_ARG_STDEV",mc_arg_sd);
	
	parseFlag("USE_HYBRID_MONTE_CARLO",use_hmc);
	if(keywords.exists("HMC_STEPS"))
		parse("HMC_STEPS",hmc_steps);
	
	if(keywords.exists("HMC_ARG_MASS"))
	{
		parseVector("HMC_ARG_MASS",hmc_arg_mass);
		if(use_hmc&&hmc_arg_mass.size()!=narg)
		{
			if(hmc_arg_mass.size()==1)
			{
				float hmass=hmc_arg_mass[0];
				hmc_arg_mass.assign(narg,hmass);
			}
			else
				plumed_merror("the number of HMC_ARG_MASS mismatch");
		}
	}
	
	parse("RANDOM_POINTS_FILE",random_file);
	random_output_steps=1;
	parse("RANDOM_OUTPUT_STEPS",random_output_steps);
	
	parse("BIAS_EPOCH_NUMBER",bias_nepoch);
	parse("BATCH_SIZE",bias_bsize);
	if(update_steps>0)
		plumed_massert(tot_md_points%bias_bsize==0,"UPDATE_STEPS must be divided exactly by BATCH_SIZE");
	fes_bsize=bias_bsize;
	
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
				if(mw_rank>0)
					random_file="/dev/null";
				orandom.enforceSuffix("");
			}
			orandom.open(random_file.c_str());
			orandom.fmtField(" %f");
		}
		
		if(nncv_bias->get_grid_wstride()>0)
		{
			std::string grid_file=nncv_bias->get_gird_file();
			if(use_mw&&mw_rank>0)
				grid_file="/dev/null";
			nncv_bias->set_gird_file(grid_file);
			nncv_bias->ogrid_init(grid_file,true);
		}
		if(nncv_fes->get_grid_wstride()>0)
		{
			std::string grid_file=nncv_fes->get_gird_file();
			if(use_mw&&mw_rank>0)
				grid_file="/dev/null";
			nncv_fes->set_gird_file(grid_file);
			nncv_fes->ogrid_init(grid_file,true);
		}
		
		bias_nbatch=tot_md_points/bias_bsize;
		fes_nbatch=tot_mc_points/fes_bsize;
		plumed_massert(tot_mc_points%(bias_nepoch*bias_nbatch)==0,"the number of MC samping points in each walker must be divided exactly by BATCH_SIZE*BIAS_EPOCH_NUMBER");
		mc_bsize=tot_mc_points/(bias_nepoch*bias_nbatch);
		log_fes_bsize=std::log(static_cast<double>(fes_bsize));
		
		md_bsize=fes_bsize;
		if(fes_bsize>tot_md_points)
			md_bsize=tot_md_points;
		md_nbatch=tot_md_points/md_bsize;
		
		for(unsigned i=0;i!=tot_md_points;++i)
			md_ids.push_back(i);
		for(unsigned i=0;i!=tot_mc_points;++i)
			mc_ids.push_back(i);
		
		addComponent("rct"); componentIsNotPeriodic("rct");
		value_rct=getPntrToComponent("rct");
		addComponent("rbias"); componentIsNotPeriodic("rbias");
		value_rbias=getPntrToComponent("rbias");
		addComponent("fes"); componentIsNotPeriodic("fes");
		value_fes=getPntrToComponent("fes");
		addComponent("Vloss"); componentIsNotPeriodic("Vloss");
		value_Vloss=getPntrToComponent("Vloss");
		addComponent("Floss"); componentIsNotPeriodic("Floss");
		value_Floss=getPntrToComponent("Floss");
	}
	
	checkRead();
	
	if(use_mw)
		log.printf("  using multiple walkers: \n");
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

NNB_Deep_MetaD::~NNB_Deep_MetaD()
{
	if(update_steps>0)
	{
		delete trainer_bias;
		delete trainer_fes;
		//~ if(is_random_ouput)
			//~ orandom.close();
	}
}

void NNB_Deep_MetaD::prepare()
{

}

void NNB_Deep_MetaD::update()
{
	
}


void NNB_Deep_MetaD::hybrid_monte_carlo(unsigned cycles)
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

void NNB_Deep_MetaD::metropolis_monte_carlo(unsigned cycles)
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

double NNB_Deep_MetaD::random_velocities(std::vector<float>& v)
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

void NNB_Deep_MetaD::random_moves(std::vector<float>& coords)
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
