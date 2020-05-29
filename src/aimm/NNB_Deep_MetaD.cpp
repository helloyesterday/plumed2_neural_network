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
#include "DynetTools.h"
#include "core/ActionRegister.h"
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
namespace aimm{
	
//+PLUMEDOC AIMM DEEP_METAD
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
	keys.add("compulsory","FES_EPOCH_NUMBER","1","the number of epoch for training the neural networks for bias potential");
	keys.add("compulsory","BIAS_BATCH_SIZE","250","batch size in each epoch for training neural networks of bias potential");
	keys.add("compulsory","FES_BATCH_SIZE","250","batch size in each epoch for training neural networks of free energy surface");
	keys.add("compulsory","BIAS_LOSS_WEIGHT","10","the weight factor of the loss function to normalize the energy of the reference structure to zero");
	keys.add("compulsory","FES_LOSS_WEIGHT","10","the weight factor of the loss function to normalize the energy of the reference structure to zero");
	keys.add("compulsory","MC_SAMPLING_POINTS","1","the sampling points for hybrid monte carlo at a MD simulation step");
	keys.add("compulsory","MC_TIMESTEP","0.001","the time step for hybrid monte carlo for the sampling of the neural network of free energy");
	keys.add("compulsory","MC_ARG_STDEV","1.0","the standard deviation of the normal distribution to generate random velocity for each argument at hybrid monte carlo");
	keys.add("compulsory","HMC_STEPS","10","the steps for a sampling of a hybrid monte carlo run");
	keys.add("compulsory","HMC_ARG_MASS","1.0","the mass for each argument at hybrid monte carlo");
	keys.add("compulsory","WT_BIAS_FACTOR","10","the well-tempered bias factor for deep metadynamics");
	
	keys.addFlag("USE_HYBRID_MONTE_CARLO",false,"use hybrid monte carlo to sample the neural networks");
	
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
	narg(nncv_ptr->get_input_number()),
	step(0),
	update_cycle(0),
	rand_prob(0,1.0),
	nnv(nncv_ptr),
	nnf(NULL),
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
	
	parse("FES_NNCV",fes_nncv_label);
	parse("ALGORITHM_BIAS",opt_bias_label);
	parse("ALGORITHM_FES",opt_fes_label);
	
	bias_scale=energy_scale;
	fes_scale=energy_scale;
	if(keywords.exists("FES_SCALE"))
	{
		float fes_factor;
		parse("FES_SCALE",fes_factor);
		fes_scale=fes_factor*kBT;
	}
	
	parse("WT_BIAS_FACTOR",wt_gamma);
	float mc_bias_factor=-1;
	parse("MC_BIAS_FACTOR",mc_bias_factor);
	parse("MC_SAMPLING_POINTS",mc_points);
	parse("MC_TIMESTEP",dt_mc);
	parseVector("MC_ARG_STDEV",mc_arg_sd);
	
	parseFlag("USE_HYBRID_MONTE_CARLO",use_hmc);
	if(keywords.exists("HMC_STEPS"))
		parse("HMC_STEPS",hmc_steps);
	if(keywords.exists("HMC_ARG_MASS"))
		parseVector("HMC_ARG_MASS",hmc_arg_mass);
		
	parse("RANDOM_POINTS_FILE",random_file);
	random_output_steps=1;
	parse("RANDOM_OUTPUT_STEPS",random_output_steps);
	
	parse("BIAS_EPOCH_NUMBER",bias_nepoch);
	parse("FES_EPOCH_NUMBER",fes_nepoch);
	
	parse("BIAS_BATCH_SIZE",bias_bsize);
	parse("FES_BATCH_SIZE",fes_bsize);
	
	parse("BIAS_LOSS_WEIGHT",bias_loss_weight);
	parse("FES_LOSS_WEIGHT",fes_loss_weight);
		
	if(update_steps>0)
	{
		sc_md_points=update_steps;
		sc_mc_points=update_steps*mc_points;
		sw_md_points=sc_md_points;
		sw_mc_points=sc_mc_points*sw_size;
		tot_md_points=sw_md_points*mw_size;
		tot_mc_points=sw_mc_points*mw_size;
		
		bias_nbatch=tot_md_points/bias_bsize;
		fes_nbatch=tot_mc_points/fes_bsize;
		plumed_massert(tot_mc_points%(bias_nepoch*bias_nbatch)==0,"the number of MC samping points in each walker must be divided exactly by BATCH_SIZE*BIAS_EPOCH_NUMBER");
		mc_bsize=tot_mc_points/(bias_nepoch*bias_nbatch);
		log_fes_bsize=std::log(static_cast<double>(fes_bsize));
		
		md_bsize=fes_bsize;
		if(fes_bsize>tot_md_points)
			md_bsize=tot_md_points;
		md_nbatch=tot_md_points/md_bsize;
		
		plumed_massert(tot_md_points%bias_bsize==0,"UPDATE_STEPS must be divided exactly by BATCH_SIZE");
		plumed_massert(wt_gamma>0,"WT_BIAS_FACTOR shoud larger than 0");
		
		mc_gamma=wt_gamma;
		
		rct_scale=(wt_gamma+1.0)/wt_gamma;
		
		for(unsigned i=0;i!=tot_md_points;++i)
			md_ids.push_back(i);
		for(unsigned i=0;i!=tot_mc_points;++i)
			mc_ids.push_back(i);

		if(mc_bias_factor>0)
		{
			use_same_gamma=false;
			mc_gamma=mc_bias_factor;
		}
		
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

		mc_start_args.resize(narg*sw_size);
		
		nnf=plumed.getActionSet().selectWithLabel<NNCV*>(fes_nncv_label);
		if(!nnf)
			plumed_merror("Neural network collective variable \""+fes_nncv_label+"\" does not exist. FES_NNCV must be NNCV type and NN_BIAS should always be defined AFTER NNCV.");
		nnf->linkBias(this);
		plumed_massert(nnf->get_input_number()==nnv->get_input_number(),"The input dimension of FES_NNCV \""+fes_nncv_label+"\" must be equal to the input of bias potential \""+get_nncv_label()+"\".");
		plumed_massert(nnf->get_output_dim()==1,"The output dimension of FES_NNCV \""+fes_nncv_label+"\" must be ONE when it used as the input of a bias potential.");
		
		if(nnv->get_grid_wstride()>0)
			nnv->set_grid_output_scale(bias_scale);
		if(nnf->get_grid_wstride()>0)
			nnf->set_grid_output_scale(fes_scale);
		
		algorithm_bias=plumed.getActionSet().selectWithLabel<Trainer_Algorithm*>(opt_bias_label);
		if(!algorithm_bias)
			plumed_merror("OPT_ALGORITHM \""+opt_bias_label+"\" does not exist. ALGORITHM_BIAS must be OPT_ALGORITHM type and DEEP_METAD should always be defined AFTER OPT_ALGORITHM.");
		algorithm_fes=plumed.getActionSet().selectWithLabel<Trainer_Algorithm*>(opt_fes_label);
		if(!algorithm_fes)
			plumed_merror("OPT_ALGORITHM \""+opt_fes_label+"\" does not exist. ALGORITHM_FES must be OPT_ALGORITHM type and DEEP_METAD should always be defined AFTER OPT_ALGORITHM.");
			
		trainer_bias=algorithm_bias->new_trainer(nnv->get_nn_model());
		trainer_fes=algorithm_fes->new_trainer(nnf->get_nn_model());
		
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
		
		if(nnv->get_grid_wstride()>0)
		{
			std::string grid_file=nnv->get_gird_file();
			if(use_mw&&mw_rank>0)
				grid_file="/dev/null";
			nnv->set_gird_file(grid_file);
			nnv->ogrid_init(grid_file,true);
		}
		if(nnf->get_grid_wstride()>0)
		{
			std::string grid_file=nnf->get_gird_file();
			if(use_mw&&mw_rank>0)
				grid_file="/dev/null";
			nnf->set_gird_file(grid_file);
			nnf->ogrid_init(grid_file,true);
		}
		
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
	
	
	if(update_steps>0)
	{
		log.printf("  with neural network collective variable as free energy surface (value network): %s.\n",fes_nncv_label.c_str());
		log.printf("  update the parameters of neural networks every %d steps.\n",int(update_steps));
		
		if(use_hmc)
			log.printf("  using hybrid monte carlo for target distribution.\n");
		else
			log.printf("  using metropolis monte carlo for target distribution.\n");
		log.printf("    with well-tempered bias factor for monte carlo: %f.\n",mc_gamma);
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
		
		log.printf("  using molecular dynamics simluation for sampled distribution.\n");
		log.printf("    with well-tempered bias factor for MD simulation: %f.\n",wt_gamma);
		log.printf("    with sampling cycle for each traning update: %d.\n",int(update_steps));
		log.printf("    with sampling points of a update cycle for each CPU core: %d.\n",int(sc_md_points));
		log.printf("    with sampling points of a update cycle for each walker: %d.\n",int(sw_md_points));
		log.printf("    with total sampling points for each update cycle: %d.\n",int(tot_md_points));
		
		log.printf("  training the free energy surface (value network) with optimization algorithm: %s.\n",opt_fes_label.c_str());
		log.printf("    with batch size for bias potential (traning for value network): %d.\n",int(md_bsize));
		log.printf("    with batch size for free energy surface (traning for value network): %d.\n",int(fes_bsize));
		log.printf("    with number of epoch of mini batches for the value networks: %d.\n",int(fes_nbatch));
		log.printf("    with weight for the loss fucntion to normalize the free energy surface: %f.\n",fes_loss_weight);
		
		log.printf("  training the free energy surface (value network) with optimization algorithm: %s.\n",opt_bias_label.c_str());
		log.printf("    with batch size for traning bias potential at political networks: %d.\n",int(bias_bsize));
		log.printf("    with batch size for traning free energy surface at political networks: %d.\n",int(mc_bsize));
		log.printf("    with number of epoch of mini batches for the political networks: %d.\n",int(bias_nbatch));
		log.printf("    with weight for the loss fucntion to normalize the bias potential energy: %f.\n",bias_loss_weight);
		
	}
}

NNB_Deep_MetaD::~NNB_Deep_MetaD()
{
	if(update_steps>0)
	{
		delete trainer_bias;
		delete trainer_fes;
		if(is_random_ouput)
			orandom.close();
	}
}

void NNB_Deep_MetaD::calculate()
{
	double voutput=getArgument(0);
	bias_now=voutput*energy_scale;
	
	setBias(bias_now);
	setOutputForce(0,-energy_scale);
	
	if(update_steps>0)
		args_now=get_cv_input();
	
	++step;
}

void NNB_Deep_MetaD::update()
{
	if(update_steps>0)
	{
		double fes=calc_fes(args_now);
		value_fes->set(fes);
		
		bias_record.push_back(bias_now);
		std::copy(args_now.begin(),args_now.end(),std::back_inserter(arg_record));

		if(step==1)
		{
			for(unsigned i=0;i!=sw_size;++i)
			{
				for(unsigned j=0;j!=narg;++j)
					mc_start_args[i*narg+j]=args_now[j];
			}
		}
		if(step%update_steps==1)
		{
			for(unsigned j=0;j!=narg;++j)
				mc_start_args[j]=args_now[j];
		}
		
		if(use_hmc)
			hybrid_monte_carlo(mc_points);
		else
			metropolis_monte_carlo(mc_points);

		if(step%update_steps==0)
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
				param_fes.resize(nnf->parameters_number());
				param_bias.resize(nnv->parameters_number());
				
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
						
						param_fes=nnf->get_parameters();
						param_bias=nnv->get_parameters();
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
				nnf->set_parameters(param_fes);
				nnv->set_parameters(param_bias);
			}
			else
			{
				if(sw_size>1)
				{
					param_fes.resize(nnf->parameters_number());
					param_bias.resize(nnv->parameters_number());
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
						param_fes=nnf->get_parameters();
						param_bias=nnv->get_parameters();
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
					
					nnf->set_parameters(param_fes);
					nnv->set_parameters(param_bias);
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
						
			value_rct->set(rct);
			value_Floss->set(floss);
			value_Vloss->set(bloss);
			bias_record.resize(0);
			arg_record.resize(0);
			fes_random.resize(0);
			arg_random.resize(0);

			++update_cycle;
		}
	}
}

float NNB_Deep_MetaD::update_fes(const std::vector<float>& old_arg_record,
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

	dynet::Expression nnf_md=nnf_output(cg,md_inputs);
	dynet::Expression nnf_mc=nnf_output(cg,mc_inputs);
	
	dynet::Expression fes_mc=fes_scale*nnf_mc;
	dynet::Expression weight_bias_mc=fes_old/mc_gamma-fes_new-dynet::transpose(fes_mc);
	dynet::Expression log_Z_mc=dynet::logsumexp_dim(beta*weight_bias_mc,0);
	
	dynet::Expression md_weights=dynet::softmax(beta*bias_inputs);
	dynet::Expression mc_weights=dynet::softmax(beta*(fes_old/mc_gamma-fes_new));
	
	dynet::Expression mean_sample=nnf_md*md_weights;
	dynet::Expression mean_target=nnf_mc*mc_weights;
	
	dynet::Expression loss_Z=0.5*dynet::squared_norm(log_Z_mc-log_fes_bsize);
	dynet::Expression loss=mean_sample-mean_target+loss_Z*fes_loss_weight;

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
		fes_trainer_update();
	}
	floss/=fes_nbatch;
	
	dynet::Dim arg_dim({narg,tot_mc_points});
	dynet::Dim e_dim({tot_mc_points});
	dynet::Expression tot_arg_random=dynet::input(cg,arg_dim,&new_arg_random);
	dynet::Expression tot_fes_old=dynet::input(cg,e_dim,&new_fes_random);
	dynet::Expression tot_fes_new=fes_output(cg,tot_arg_random);
	
	update_fes_random=dynet::as_vector(cg.forward(tot_fes_new));
	
	dynet::Expression m_beta_F=beta*(tot_fes_old/mc_gamma-2*dynet::transpose(tot_fes_new));
	dynet::Expression tot_log_Z=dynet::logsumexp_dim(m_beta_F,0);
	std::vector<float> out=dynet::as_vector(cg.forward(tot_log_Z));
	log_Z=out[0];
	
	return floss;
}

float NNB_Deep_MetaD::update_bias(const std::vector<float>& vec_arg_record,
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
	
	dynet::Expression nnv_mc=nnv_output(cg,mc_inputs);
	dynet::Expression bias_mc=dynet::transpose(bias_scale*nnv_mc);
	dynet::Expression nnv_md=nnv_output(cg,md_inputs);
	
	dynet::Expression log_wbias_mc=fes_old/mc_gamma-fes_new/wt_gamma;
	
	dynet::Expression log_m_F_p_V=fes_old/mc_gamma-fes_new*rct_scale-bias_mc;
	dynet::Expression log_Zb=dynet::logsumexp_dim(beta*log_m_F_p_V,0);
	
	dynet::Expression dy_rct=log_Z-log_Zb;
	
	dynet::Expression loss_rct=0.5*dynet::squared_norm(dy_rct);
	
	dynet::Expression mc_weights=dynet::softmax(beta*log_wbias_mc);
	dynet::Expression mean_target=nnv_mc*mc_weights;
	
	dynet::Expression mean_sample;
	
	dynet::Expression bias_md=bias_scale*nnv_md;
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
	
	dynet::Expression loss=mean_target-mean_sample+loss_rct*bias_loss_weight;
	
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
			
			bias_trainer_update();
		}
	}
	
	bloss/=bias_nbatch*bias_nepoch;
	
	dynet::Dim arg_dim({narg,tot_mc_points});
	dynet::Dim e_dim({tot_mc_points});
	dynet::Expression arg_input=dynet::input(cg,arg_dim,&vec_arg_random);
	dynet::Expression tot_fes_old=dynet::input(cg,e_dim,&origin_fes_random);
	dynet::Expression tot_fes_new=dynet::input(cg,e_dim,&update_fes_random);
	dynet::Expression tot_bias_new=dynet::transpose(bias_output(cg,arg_input));
	dynet::Expression fin_log_m_F_p_V=tot_fes_old/mc_gamma-tot_fes_new*rct_scale-tot_bias_new;
	dynet::Expression fin_log_Zb=dynet::logsumexp_dim(beta*fin_log_m_F_p_V,0);
	
	std::vector<float> out=dynet::as_vector(cg.forward(fin_log_Zb));
	rct=(log_Z-out[0])/beta;
	
	return bloss;
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
	dynet::Expression fes=fes_output(cg,inputs);
	
	std::vector<float> deriv;
	std::vector<float> vf0=calc_output_and_gradient(cg,inputs,fes,deriv);
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
			std::vector<float> vf1=calc_output_and_gradient(cg,inputs,fes,dF_dr1);
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
			double acprob=exp(-beta*(H1-H0)/mc_gamma);
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
	dynet::Expression fes=fes_output(cg,inputs);
	
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
			double acprob=exp(-beta*(E1-E0)/mc_gamma);
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
		if(arg_is_periodic(i))
		{
			if(coords[i]<arg_min(i))
				coords[i]+=arg_period(i);
			else if(coords[i]>arg_max(i))
				coords[i]-=arg_period(i);
		}
	}
}

float NNB_Deep_MetaD::calc_energy(dynet::ComputationGraph& cg,const std::vector<float>& cvs,NNCV* nncv)
{
	dynet::Expression inputs=dynet::input(cg,{narg},&cvs);
	dynet::Expression output=nncv->output(cg,inputs);
	std::vector<float> out=calc_output(cg,inputs,output);
	return out[0];
}

float NNB_Deep_MetaD::calc_energy(const std::vector<float>& cvs,NNCV* nncv)
{
	dynet::ComputationGraph cg;
	return calc_energy(cg,cvs,nncv);
}

float NNB_Deep_MetaD::calc_energy_and_deriv(dynet::ComputationGraph& cg,const std::vector<float>& cvs,std::vector<float>& deriv,NNCV* nncv)
{
	dynet::Expression inputs=dynet::input(cg,{narg},&cvs);
	dynet::Expression output=nncv->output(cg,inputs);
	std::vector<float> out=calc_output_and_gradient(cg,inputs,output,deriv);
	return out[0];
}

float NNB_Deep_MetaD::calc_energy_and_deriv(const std::vector<float>& cvs,std::vector<float>& deriv,NNCV* nncv)
{
	dynet::ComputationGraph cg;
	return calc_energy_and_deriv(cg,cvs,deriv,nncv);
}


}
}

#endif
