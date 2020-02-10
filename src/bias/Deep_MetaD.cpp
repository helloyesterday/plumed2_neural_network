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
#include <string>
#include <cstring>
#include "tools/File.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <ctime>
#include <cmath>

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
	keys.add("compulsory","BIAS_LAYERS_NUMBER","3","the hidden layers of the multilayer preceptron");
	keys.add("compulsory","BIAS_LAYER_DIMENSIONS","64","the dimension of each hidden layers");
	keys.add("compulsory","BIAS_ACTIVE_FUNCTIONS","SWISH","the activation function for the neural network");
	keys.add("compulsory","FES_LAYERS_NUMBER","3","the hidden layers of the multilayer preceptron");
	keys.add("compulsory","FES_LAYER_DIMENSIONS","64","the dimension of each hidden layers");
	keys.add("compulsory","FES_ACTIVE_FUNCTIONS","SWISH","the activation function for the neural network");
	keys.add("compulsory","CLIP_RANGE","-0.01,0.01","the range of the value to clip");
	keys.add("compulsory","SCALE_FACTOR","5","a constant to scale the output of neural network as bias potential");
	keys.add("compulsory","BIAS_PARAM_OUTPUT","bias_parameters.data","the file to output the parameters of the neural network");
	keys.add("compulsory","FES_PARAM_OUTPUT","fes_parameters.data","the file to output the parameters of the neural network");
	keys.add("compulsory","HMC_STEPS","100","the steps for hybrid monte carlo at a update cycle");
	keys.add("compulsory","HMC_TIMESTEP","0.01","the time step for hybrid monte carlo for the sampling of the neural network of free energy");
	keys.add("compulsory","HMC_ARG_MASS","1.0","the mass for each argument at hybrid monte carlo");
	keys.add("compulsory","HMC_ARG_STDEV","1.0","the standard deviation of the normal distribution to generate random velocity for each argument at hybrid monte carlo");
	
	keys.add("optional","BIAS_ALGORITHM","(default=ADAM) the algorithm to train the discriminator (value network)");
	keys.add("optional","FES_ALGORITHM","(default=ADAM) the algorithm to train the discriminator (value network)");
	keys.add("optional","BIAS_LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","BIAS_HYPER_PARAMS","other hyperparameters for training the neural network");
	keys.add("optional","BIAS_CLIP_THRESHOLD","the clip threshold for training the neural network");
	keys.add("optional","FES_LEARN_RATE","the learning rate for training the neural network");
	keys.add("optional","FES_HYPER_PARAMS","other hyperparameters for training the neural network");
	keys.add("optional","FES_CLIP_THRESHOLD","the clip threshold for training the neural network");
	keys.add("optional","BIAS_PARAM_READ_FILE","the file to output the parameters of the neural network");
	keys.add("optional","FES_PARAM_READ_FILE","the file to output the parameters of the neural network");
	keys.add("optional","SIM_TEMP","the simulation temerature");
	
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
}

Deep_MetaD::Deep_MetaD(const ActionOptions& ao):
	PLUMED_BIAS_INIT(ao),
	narg(getNumberOfArguments()),
	steps(0),
	firsttime(true),
	rand_prob(0,1.0),
	trainer_bias(NULL),
	trainer_fes(NULL)
{
	parse("BIAS_LAYERS_NUMBER",nlv);
	plumed_massert(nlv>0,"LAYERS_NUMBER must be larger than 0!");
	parse("FES_LAYERS_NUMBER",nlf);
	plumed_massert(nlv>0,"LAYERS_NUMBER must be larger than 0!");

	parseVector("BIAS_LAYER_DIMENSIONS",ldv);
	if(ldv.size()!=nlv)
	{
		if(ldv.size()==1)
		{
			unsigned dim=ldv[0];
			ldv.assign(nlv,dim);
		}
		else
			plumed_merror("BIAS_LAYER_DIMENSIONS mismatch!");
	}
	
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

	std::vector<std::string> safv;
	parseVector("BIAS_ACTIVE_FUNCTIONS",safv);
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
	afv.resize(nlv);
	for(unsigned i=0;i!=nlv;++i)
		afv[i]=activation_function(safv[i]);

	std::vector<std::string> saff;
	parseVector("BIAS_ACTIVE_FUNCTIONS",saff);
	if(saff.size()!=nlf)
	{
		if(saff.size()==1)
		{
			std::string af=saff[0];
			saff.assign(nlf,af);
		}
		else
			plumed_merror("ACTIVE_FUNCTIONS mismatch!");
	}
	aff.resize(nlf);
	for(unsigned i=0;i!=nlf;++i)
		aff[i]=activation_function(saff[i]);

	parse("BIAS_PARAM_OUTPUT",bias_file_out);
	parse("FES_PARAM_OUTPUT",fes_file_out);
	parse("BIAS_PARAM_READ_FILE",bias_file_in);
	parse("FES_PARAM_READ_FILE",fes_file_in);
	
	parse("SCALE_FACTOR",scale_factor);
	
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
	
	bias_scale=scale_factor*kBT;
	
	parse("UPDATE_STEPS",update_steps);
	
	parseFlag("MULTIPLE_WALKERS",use_mw);
	
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
		bias_algorithm="ADAM";
		parse("BIAS_ALGORITHM",bias_algorithm);
		parse("BIAS_CLIP_THRESHOLD",bias_clip_threshold);
		parseVector("BIAS_LEARN_RATE",lrv);
		std::vector<float> opv;
		parseVector("BIAS_HYPER_PARAMS",opv);
		
		fes_algorithm="ADAM";
		parse("FES_ALGORITHM",fes_algorithm);
		parse("FES_CLIP_THRESHOLD",fes_clip_threshold);
		parseVector("FES_LEARN_RATE",lrf);
		std::vector<float> opf;
		parseVector("FES_HYPER_PARAMS",opf);
		
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
		unsigned ldim=narg;
		for(unsigned i=0;i!=nlv;++i)
		{
			nnv.append(pcv,Layer(ldim,ldv[i],afv[i],0));
			ldim=ldv[i];
		}
		nnv.append(pcv,Layer(ldim,1,Activation::LINEAR,0));
		
		ldim=narg;
		for(unsigned i=0;i!=nlf;++i)
		{
			nnf.append(pcf,Layer(ldim,ldf[i],aff[i],0));
			ldim=ldv[i];
		}
		nnf.append(pcf,Layer(ldim,1,Activation::LINEAR,0));
		
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
	
	if(update_steps>0&&steps>0&&(steps%update_steps==0))
	{
		update_fes();
		update_bias();
	}

	std::vector<float> deriv;
	double bias_pot=calc_energy(args,deriv);
	
	setBias(bias_pot);
	
	for(unsigned i=0;i!=narg;++i)
	{
		double bias_force=-deriv[i];
		setOutputForce(i,bias_force);
	}
	
	++steps;
}

float Deep_MetaD::calc_energy(const std::vector<float>& args,std::vector<float>& deriv)
{
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&args);
	dynet::Expression output=bias_scale*nnv.run(inputs,cg);
	
	cg.forward(output);
	std::vector<float> outvec=dynet::as_vector(output.value());
	
	cg.backward(output,true);
	deriv=dynet::as_vector(inputs.gradient());
	
	return outvec[0];
}

float Deep_MetaD::update_fes()
{
	dynet::ComputationGraph cg;
	dynet::Dim in_dim({narg},update_steps);
	dynet::Expression inputs=dynet::input(cg,in_dim,&arg_record);
	
	dynet::Expression fes_md=bias_scale*nnf.run(inputs,cg);
	dynet::Expression md_mean=dynet::mean_batches(fes_md);
	
	dynet::Expression loss1=md_mean;
	float vloss1 = as_scalar(cg.forward(loss1));
	cg.backward(loss1);
	trainer_fes->update();
	
	return vloss1;
}

float Deep_MetaD::update_bias()
{
	dynet::ComputationGraph cg;
	dynet::Dim in_dim({narg},update_steps);
	dynet::Expression inputs=dynet::input(cg,in_dim,&arg_record);
	
	dynet::Expression bias_md=nnv.run(inputs,cg);
	dynet::Expression md_mean=dynet::mean_batches(bias_md);
	
	dynet::Expression loss1=md_mean;
	float vloss1 = as_scalar(cg.forward(loss1));
	cg.backward(loss1);
	trainer_bias->update();
	
	arg_record.resize(0);
	
	return vloss1;
}

std::vector<float> Deep_MetaD::hybrid_monte_carlo(const std::vector<float>& init_coords)
{
	std::vector<float> v0(narg);
	double K0=random_velocities(v0);
		
	std::vector<float> r0(init_coords);
	std::vector<float> r_input(r0);
	
	dynet::ComputationGraph cg;
	dynet::Expression inputs=dynet::input(cg,{narg},&r_input);
	dynet::Expression fes=bias_scale*nnf.run(inputs,cg);
	
	std::vector<float> deriv0;
	double F0=get_output_and_gradient(cg,inputs,fes,deriv0);
	double H0=F0+K0;
	std::back_insert_iterator<std::vector<float>> back_iter(arg_random);
	std::copy(r0.begin(),r0.end(),back_iter);
	
	for(unsigned i=1;i!=hmc_steps;++i)
	{
		std::vector<float> vh(narg);
		std::vector<float> r1(narg);
		for(unsigned j=0;j!=narg;++j)
		{
			double v=v0[j]-deriv0[j]*dt_hmc/hmc_arg_mass[j]/2;
			vh[j]=v;
			double r=r0[j]+v*dt_hmc;
			r1[j]=r;
			r_input[j]=r;
		}
		
		std::vector<float> deriv1;
		double F1=get_output_and_gradient(cg,inputs,fes,deriv1);

		double K1=0;
		std::vector<float> v1(narg);
		for(unsigned j=0;j!=narg;++j)
		{
			double v=vh[j]-deriv1[j]*dt_hmc/hmc_arg_mass[j]/2;
			v1[j]=v;
			K1+=hmc_arg_mass[j]*v*v/2;
		}
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
			F0=F1;
			H0=H1;
			r0.swap(r1);
			v0.swap(v1);
			deriv0.swap(deriv1);
		}
		else
		{
			K0=random_velocities(v0);
			H0=F0+K0;
		}
		
		std::copy(r0.begin(),r0.end(),back_iter);
	}
	
	return r0;
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

double Deep_MetaD::get_output_and_gradient(dynet::ComputationGraph& cg,dynet::Expression& inputs,dynet::Expression& output,std::vector<float>& deriv)
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
