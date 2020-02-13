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
	keys.remove("ARG");
    keys.add("optional","ARG","the argument(CVs) here must be the potential energy. If no argument is setup, only bias will be integrated in this method."); 
	//~ keys.use("ARG");

	keys.add("compulsory","UPDATE_STEPS","1000","the frequency to update parameters of neural networks. ZERO means do not update the parameters");
	keys.add("compulsory","LAYERS_NUMBER","3","the hidden layers of the multilayer preceptron of the neural networks");
	keys.add("compulsory","LAYER_DIMENSIONS","64","the dimension of each hidden layers of the neural networks");
	keys.add("compulsory","ACTIVE_FUNCTIONS","SWISH","the activation function for the neural networks");
	keys.add("compulsory","OPT_ALGORITHM","ADAM","the algorithm to train the neural networks");
	keys.add("compulsory","CLIP_RANGE","-0.01,0.01","the range of the value to clip");
	keys.add("compulsory","SCALE_FACTOR","5","a constant to scale the output of neural network as bias potential");
	keys.add("compulsory","BIAS_PARAM_OUTPUT","bias_parameters.data","the file to output the parameters of the neural network of the bias potential");
	keys.add("compulsory","FES_PARAM_OUTPUT","fes_parameters.data","the file to output the parameters of the neural network of the free energy surface");
	keys.add("compulsory","HMC_SAMPLING_POINTS","100","the sampling points for hybrid monte carlo at a MD simulation step");
	keys.add("compulsory","HMC_STEPS","10","the steps for a sampling of a hybrid monte carlo run");
	keys.add("compulsory","HMC_TIMESTEP","0.001","the time step for hybrid monte carlo for the sampling of the neural network of free energy");
	keys.add("compulsory","HMC_ARG_MASS","1.0","the mass for each argument at hybrid monte carlo");
	keys.add("compulsory","HMC_ARG_STDEV","1.0","the standard deviation of the normal distribution to generate random velocity for each argument at hybrid monte carlo");
	keys.add("compulsory","WT_BIAS_FACTOR","10","the well-tempered bias factor for deep metadynamics");
	
	keys.addFlag("USE_DIFF_PARAM_FOR_FES",false,"use different parameters for the neural network of free energy surface");
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
	
	keys.add("optional","LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","HYPER_PARAMS","other hyperparameters for training the neural network");
	keys.add("optional","CLIP_THRESHOLD","the clip threshold for training the neural network");
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
	firsttime(true),
	rand_prob(0,1.0),
	arg_pbc(getNumberOfArguments(),false),
	arg_init(getNumberOfArguments(),0),
	arg_min(getNumberOfArguments(),0),
	arg_max(getNumberOfArguments(),0),
	trainer_bias(NULL),
	trainer_fes(NULL)
{
	std::vector<std::string> arg_label(narg);
	is_arg_has_pbc=false;
	for(unsigned i=0;i!=narg;++i)
	{
		arg_pbc[i]=getPntrToArgument(i)->isPeriodic();
		arg_label[i]=getNumberOfArguments();
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
			plumed_merror("LAYER_DIMENSIONS mismatch!");
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
			plumed_merror("ACTIVE_FUNCTIONS mismatch!");
	}
	std::vector<std::string> saff=safv;
	
	afv.resize(nlv);
	for(unsigned i=0;i!=nlv;++i)
		afv[i]=activation_function(safv[i]);
	aff=afv;
	
	if(use_diff_param)
	{
		parse("FES_LAYERS_NUMBER",nlf);
		plumed_massert(nlv>0,"LAYERS_NUMBER must be larger than 0!");
		parseVector("FES_LAYER_DIMENSIONS",ldf);
		if(ldf.size()!=nlf)
		{
			if(ldf.size()==1)
			{
				unsigned dim=ldf[0];
				ldf.assign(nlf,dim);
			}
			else
				plumed_merror("FES_LAYER_DIMENSIONS mismatch!");
		}
		
		saff.resize(0);
		parseVector("FES_ACTIVE_FUNCTIONS",saff);
		if(saff.size()!=nlf)
		{
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

	parse("BIAS_PARAM_OUTPUT",bias_file_out);
	parse("FES_PARAM_OUTPUT",fes_file_out);
	parse("BIAS_PARAM_READ_FILE",bias_file_in);
	parse("FES_PARAM_READ_FILE",fes_file_in);
	
	parse("SCALE_FACTOR",scale_factor);
	parse("WT_BIAS_FACTOR",bias_factor);
	bias_scale=1.0-1.0/bias_factor;
	
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
	
	parse("UPDATE_STEPS",update_steps);
	
	parseFlag("MULTIPLE_WALKERS",use_mw);
	
	parse("HMC_SAMPLING_POINTS",hmc_points);
	parse("HMC_STEPS",hmc_steps);
	parse("HMC_TIMESTEP",dt_hmc);
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
	
	parseVector("HMC_ARG_STDEV",hmc_arg_sd);
	if(hmc_arg_sd.size()!=narg)
	{
		if(hmc_arg_sd.size()==1)
		{
			float hsd=hmc_arg_sd[0];
			hmc_arg_sd.assign(narg,hsd);
		}
		else
			plumed_merror("the number of HMC_ARG_STDEV mismatch");
	}
	for(unsigned i=0;i!=narg;++i)
		ndist.push_back(std::normal_distribution<float>(0,hmc_arg_sd[i]));

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
	if(update_steps>0)
	{	
		parse("OPT_ALGORITHM",bias_algorithm);
		fes_algorithm=bias_algorithm;
		
		parse("CLIP_THRESHOLD",bias_clip_threshold);
		fes_clip_threshold=bias_clip_threshold;
		
		parseVector("LEARN_RATE",lrv);
		lrf=lrv;

		std::vector<float> opv;
		parseVector("HYPER_PARAMS",opv);
		std::vector<float> opf=opv;
		
		if(use_diff_param)
		{
			parse("FES_ALGORITHM",fes_algorithm);
			parse("FES_CLIP_THRESHOLD",fes_clip_threshold);
			parseVector("FES_LEARN_RATE",lrf);
			parseVector("FES_HYPER_PARAMS",opf);
		}
		
		std::vector<float> clips; 
		parseVector("CLIP_RANGE",clips);
		plumed_massert(clips.size()>=2,"CLIP_RANGE should has left and right values");
		clip_left=clips[0];
		clip_right=clips[1];
		plumed_massert(clip_right>clip_left,"Clip left value should less than clip right value");
		
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
		
		trainer_bias->clip_threshold = bias_clip_threshold;
		trainer_fes->clip_threshold = fes_clip_threshold;
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
	
	log.printf("  with neural network for bias potential:\n");
	log.printf("    with number of hidden layers: %d\n",int(nlv));
	for(unsigned i=0;i!=nlv;++i)
		log.printf("      Hidden layer %d with dimension %d and activation funciton %s\n",int(i),int(ldv[i]),safv[i].c_str());
	log.printf("    with parameters output file: %s\n",bias_file_out.c_str());
	if(bias_file_in.size()>0)
		log.printf("    with parameters read file: %s\n",bias_file_in.c_str());
	if(update_steps>0)
		log.printf("    update the parameters of neural networks using %s algorithm.\n",bias_fullname.c_str());
		
	log.printf("  with neural network for free energy:\n");
	log.printf("    with number of hidden layers: %d\n",int(nlf));
	for(unsigned i=0;i!=nlf;++i)
		log.printf("      Hidden layer %d with dimension %d and activation funciton %s\n",int(i),int(ldf[i]),saff[i].c_str());
	log.printf("    with parameters output file: %s\n",fes_file_out.c_str());
	if(fes_file_in.size()>0)
		log.printf("    with parameters read file: %s\n",fes_file_in.c_str());
	if(update_steps>0)
		log.printf("    update the parameters of neural networks using %s algorithm.\n",fes_fullname.c_str());
	
	if(update_steps==0)
		log.printf("    without updating the parameters of neural networks.\n");
	else
		log.printf("  update the parameters of neural networks every %d steps.\n",int(update_steps));
}

Deep_MetaD::~Deep_MetaD()
{
	if(update_steps>0)
	{
		delete trainer_bias;
		delete trainer_fes;
	}
}


void Deep_MetaD::prepare()
{
	if(firsttime)
	{
		nnv.set_hidden_layers(ldv,afv);
		nnv.build_neural_network(pcv);
		nnf.set_hidden_layers(ldf,aff);
		nnf.build_neural_network(pcf);
		
		if(bias_file_in.size()>0)
		{
			dynet::TextFileLoader loader(bias_file_in);
			loader.populate(pcv);
		}
		dynet::TextFileSaver savev(bias_file_out);
		savev.save(pcv);
		
		if(fes_file_in.size()>0)
		{
			dynet::TextFileLoader loader(fes_file_in);
			loader.populate(pcf);
		}
		dynet::TextFileSaver savef(fes_file_out);
		savef.save(pcf);
		
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
		if(update_steps>0)
			arg_record.push_back(val);
	}

	std::vector<float> deriv;
	double bias_pot=calc_energy(args,deriv);
	
	setBias(bias_pot);
	
	for(unsigned i=0;i!=narg;++i)
	{
		double bias_force=-deriv[i];
		setOutputForce(i,bias_force);
	}
	
	if(update_steps>0)
	{
		bias_record.push_back(bias_pot);
		std::copy(args.begin(),args.end(),
			std::back_insert_iterator<std::vector<float>>(arg_record));

		if(steps==0)
			arg_init=args;
		hybrid_monte_carlo(arg_init);

		if(steps>0&&(steps%update_steps==0))
		{
			update_fes();
			update_bias();
			bias_record.resize(0);
			arg_record.resize(0);
			fes_random.resize(0);
			arg_random.resize(0);
			arg_init.swap(args);
		}
	}
	
	++steps;
}

float Deep_MetaD::calc_energy(const std::vector<float>& args,std::vector<float>& deriv)
{
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&args);
	dynet::Expression output=nnv.energy(cg,inputs);
	return get_output_and_gradient(cg,inputs,output,deriv);
}

float Deep_MetaD::update_fes()
{
	dynet::ComputationGraph cg;
	
	dynet::Dim bias_dim({1},update_steps);
	dynet::Dim md_dim({narg},update_steps);
	dynet::Dim mc_dim({narg},update_steps*hmc_steps);
	
	dynet::Expression md_bias=dynet::input(cg,bias_dim,&arg_random);
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record);
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random);
	
	dynet::Expression fes_md=nnf.energy(cg,md_inputs)*dynet::exp(beta*md_bias);
	dynet::Expression fes_mc=nnf.energy(cg,mc_inputs);
	
	dynet::Expression md_mean=dynet::mean_batches(fes_md);
	dynet::Expression mc_mean=dynet::mean_batches(fes_mc);
	
	dynet::Expression loss=md_mean-mc_mean;
	float vloss = as_scalar(cg.forward(loss));
	cg.backward(loss);
	trainer_fes->update();
	
	return vloss;
}

float Deep_MetaD::update_bias()
{
	dynet::ComputationGraph cg;
	
	dynet::Dim md_dim({narg},update_steps);
	dynet::Dim mc_dim({narg},update_steps*hmc_steps);
	dynet::Dim fes_dim({1},update_steps*hmc_steps);
	
	dynet::Expression md_inputs=dynet::input(cg,md_dim,&arg_record);
	dynet::Expression mc_inputs=dynet::input(cg,mc_dim,&arg_random);
	dynet::Expression mc_fes=dynet::input(cg,fes_dim,&fes_random);
	
	dynet::Expression bias_md=nnv.energy(cg,md_inputs);
	dynet::Expression bias_mc=nnv.energy(cg,mc_inputs)*dynet::exp(beta*bias_scale*mc_fes);
	
	dynet::Expression md_sample=dynet::mean_batches(bias_md);
	dynet::Expression md_target=dynet::mean_batches(bias_mc);
	
	dynet::Expression loss=md_target-md_sample;
	float vloss = as_scalar(cg.forward(loss));
	cg.backward(loss);
	trainer_bias->update();
	
	arg_record.resize(0);
	
	return vloss;
}

void Deep_MetaD::hybrid_monte_carlo(std::vector<float>& init_coords)
{
	std::vector<float> coords(init_coords);
	std::vector<float> r_input(coords);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r_input);
	dynet::Expression fes=nnf.energy(cg,inputs);
	
	std::vector<float> deriv;
	double F0=get_output_and_gradient(cg,inputs,fes,deriv);
	fes_random.push_back(F0);
	std::back_insert_iterator<std::vector<float>> back_iter(arg_random);
	std::copy(coords.begin(),coords.end(),back_iter);
	
	for(unsigned i=1;i!=hmc_points;++i)
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
				double v=v0[k]-dF_dr0[k]*dt_hmc/hmc_arg_mass[k]/2;
				vh[k]=v;
				double r=r0[k]+v*dt_hmc;
				r1[k]=r;
				r_input[k]=r;
			}
			
			std::vector<float> dF_dr1;
			F1=get_output_and_gradient(cg,inputs,fes,dF_dr1);
			
			std::vector<float> v1(narg);
			for(unsigned k=0;k!=narg;++k)
				v1[k]=vh[k]-dF_dr1[k]*dt_hmc/hmc_arg_mass[k]/2;
			
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
		std::copy(coords.begin(),coords.end(),back_iter);
	}
	
	init_coords.swap(coords);
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

float Deep_MetaD::get_output_and_gradient(dynet::ComputationGraph& cg,dynet::Expression& inputs,dynet::Expression& output,std::vector<float>& deriv)
{
	cg.forward(output);
	cg.backward(output,true);
	std::vector<float> out=dynet::as_vector(output.value());
	deriv=dynet::as_vector(inputs.gradient());
	return out[0];
}

}
}

#endif
