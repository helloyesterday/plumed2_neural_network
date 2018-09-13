/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The VES code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of VES code module.

   The VES code module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The VES code module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the VES code module.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_DYNET

#include "Optimizer.h"
#include "CoeffsVector.h"
#include "CoeffsMatrix.h"

#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace ves {

using namespace dytools;

//+PLUMEDOC VES_OPTIMIZER OPT_TALOS
/*
Targeted Adversarial Learning Optimizied Sampling (TALOS)

An optimization algorithm for VES using Wasserstein ganerative adversarial network (W-GAN)

\par Algorithm

\par Examples

\endplumedfile

\endplumedfile

*/
//+ENDPLUMEDOC

class Opt_TALOS :
	public Optimizer,
	public ActionWithArguments
{
private:
	bool is_debug;
	bool opt_const;
	
	unsigned random_seed;
	
	unsigned narg;
	unsigned nbiases;
	unsigned tot_basis;
	unsigned ntarget;
	unsigned nepoch;
	unsigned batch_size;
	unsigned step;
	unsigned counts;
	unsigned update_steps;
	unsigned nw;
	unsigned iter_time;
	
	double kB;
	double kBT;
	double beta;
	double sim_temp;
	
	double clip_threshold_bias;
	double clip_threshold_wgan;
	
	float clip_left;
	float clip_right;
	
	std::vector<unsigned> basises_nums;
	std::vector<unsigned> const_id;

	std::vector<bool> args_periodic;

	std::vector<double> grid_min;
	std::vector<double> grid_max;
	std::vector<unsigned> grid_bins;
	std::vector<float> grid_space;

	std::vector<float> input_arg;
	std::vector<float> input_bias;
	
	std::vector<dynet::real> input_target;
	std::vector<dynet::real> p_target;
	std::vector<dynet::real> basis_values;
	
	std::string algorithm_bias;
	std::string algorithm_wgan;
	
	std::string debug_file;
	OFile odebug;
	
	std::vector<float> lr_bias;
	std::vector<float> lr_wgan;
	std::vector<float> hyper_params_bias;
	std::vector<float> hyper_params_wgan;
	
	ParameterCollection pc_bias;
	ParameterCollection pc_wgan;
	
	Trainer *train_bias;
	Trainer *train_wgan;
	
	Parameter parm_bias;
	MLP nn_wgan;
	
	Value* valueLwgan;
	Value* valueLbias;
	
	float update_wgan(const std::vector<float>&,std::vector<std::vector<float>>&);
	float update_bias(const std::vector<float>&,const std::vector<std::vector<float>>&);
	
	Trainer* new_traniner(const std::string&,ParameterCollection&,std::string&);
	Trainer* new_traniner(const std::string&,ParameterCollection&,const std::vector<float>&,std::string&);
public:
	static void registerKeywords(Keywords&);
	explicit Opt_TALOS(const ActionOptions&);
	~Opt_TALOS();
	void update();
	void coeffsUpdate(const unsigned int c_id = 0){}
	template<class T>
	bool parseVectorAuto(const std::string&, std::vector<T>&,unsigned);
};

PLUMED_REGISTER_ACTION(Opt_TALOS,"OPT_TALOS")

void Opt_TALOS::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);
  Optimizer::useFixedStepSizeKeywords(keys);
  Optimizer::useMultipleWalkersKeywords(keys);
  Optimizer::useHessianKeywords(keys);
  Optimizer::useMaskKeywords(keys);
  Optimizer::useRestartKeywords(keys);
  Optimizer::useDynamicTargetDistributionKeywords(keys);
  ActionWithArguments::registerKeywords(keys);
  keys.addOutputComponent("wloss","default","loss function of the W-GAN");
  keys.addOutputComponent("bloss","default","loss function of the neural network of bias function");
  keys.use("ARG");
  //~ keys.remove("ARG");
  //~ keys.add("compulsory","ARG","the arguments used to set the target distribution");
  keys.remove("STRIDE");
  keys.remove("STEPSIZE");
  keys.add("compulsory","ARG_PERIODIC","NO","if the arguments are periodic or not");
  keys.add("compulsory","ALGORITHM_BIAS","ADAM","the algorithm to train the neural network of bias function");
  keys.add("compulsory","ALGORITHM_WGAN","ADAM","the algorithm to train the W-GAN");
  keys.add("compulsory","UPDATE_STEPS","250","the number of step to update");
  keys.add("compulsory","EPOCH_NUM","1","number of epoch for each update per walker");
  keys.add("compulsory","HIDDEN_NUMBER","3","the number of hidden layers for W-GAN");
  keys.add("compulsory","HIDDEN_LAYER","8","the dimensions of each hidden layer  for W-GAN");
  keys.add("compulsory","HIDDEN_ACTIVE","RELU","active function of each hidden layer  for W-GAN");
  keys.add("compulsory","CLIP_LEFT","-0.01","the left value to clip");
  keys.add("compulsory","CLIP_RIGHT","0.01","the right value to clip");
  keys.add("compulsory","GRID_MIN","the lower bounds used to calculate the target distribution");
  keys.add("compulsory","GRID_MAX","the upper bounds used to calculate the target distribution");
  keys.add("compulsory","GRID_BINS","the number of bins used to set the target distribution");
  keys.addFlag("OPTIMIZE_CONSTANT_PARAMETER",false,"also to optimize the constant part of the basis functions");
  keys.add("optional","LEARN_RATE_BIAS","the learning rate for training the neural network of bias function");
  keys.add("optional","LEARN_RATE_WGAN","the learning rate for training the W-GAN");
  keys.add("optional","HYPER_PARAMS_BIAS","other hyperparameters for training the neural network of bias function");
  keys.add("optional","HYPER_PARAMS_WGAN","other hyperparameters for training the W-GAn");
  keys.add("optional","CLIP_THRESHOLD_BIAS","the clip threshold for training the neural network of bias function");
  keys.add("optional","CLIP_THRESHOLD_WGAN","the clip threshold for training the W-GAn");
  keys.add("optional","SIM_TEMP","the simulation temperature");
  keys.add("optional","DEBUG_FILE","the file to debug");
  keys.add("hidden","COMBINED_GRADIENT_FILE","the name of output file for the combined gradient (gradient + Hessian term)");
  keys.add("hidden","COMBINED_GRADIENT_OUTPUT","how often the combined gradient should be written to file. This parameter is given as the number of bias iterations. It is by default 100 if COMBINED_GRADIENT_FILE is specficed");
  keys.add("hidden","COMBINED_GRADIENT_FMT","specify format for combined gradient file(s) (useful for decrease the number of digits in regtests)");
  keys.add("optional","EXP_DECAYING_AVER","calculate the averaged coefficients using exponentially decaying averaging using the decaying constant given here in the number of iterations");
}


Opt_TALOS::~Opt_TALOS() {
	delete train_wgan;
	delete train_bias;
	if(is_debug)
		odebug.close();
}


Opt_TALOS::Opt_TALOS(const ActionOptions&ao):
  PLUMED_VES_OPTIMIZER_INIT(ao),
  ActionWithArguments(ao),
  is_debug(false),
  random_seed(0),
  narg(getNumberOfArguments()),
  step(0),
  counts(0),
  nw(1),
  iter_time(0),
  args_periodic(getNumberOfArguments(),false),
  grid_space(getNumberOfArguments()),
  train_bias(NULL),
  train_wgan(NULL),
  nn_wgan(pc_wgan)
{
	int cc=1;
	char pp[]="plumed";
	char *vv[]={pp};
	char** ivv=vv;
	DynetParams params = extract_dynet_params(cc,ivv);
	if(useMultipleWalkers())
	{
		if(multi_sim_comm.Get_rank()==0)
		{
			if(comm.Get_rank()==0)
			{
				dynet::initialize(params);
				random_seed=params.random_seed;
			}
			comm.Barrier();
			comm.Bcast(random_seed,0);
			params.random_seed=random_seed;
			if(comm.Get_rank()!=0)
				dynet::initialize(params);
		}
		multi_sim_comm.Barrier();
		multi_sim_comm.Bcast(random_seed,0);
		comm.Bcast(random_seed,0);
		params.random_seed=random_seed;
		if(multi_sim_comm.Get_rank()!=0)
			dynet::initialize(params);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			dynet::initialize(params);
			random_seed=params.random_seed;
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);
		params.random_seed=random_seed;
		if(comm.Get_rank()!=0)
			dynet::initialize(params);
	}
	
	setStride(1);
	parse("UPDATE_STEPS",update_steps);
	
	parse("ALGORITHM_BIAS",algorithm_bias);
	parse("ALGORITHM_WGAN",algorithm_wgan);
	
	parseVector("LEARN_RATE_BIAS",lr_bias);
	parseVector("LEARN_RATE_WGAN",lr_wgan);
	
	std::vector<float> other_params_bias,other_params_wgan;
	parseVector("HYPER_PARAMS_BIAS",other_params_bias);
	parseVector("HYPER_PARAMS_BIAS",other_params_wgan);
	
	unsigned nhidden=0;
	parse("HIDDEN_NUMBER",nhidden);
	
	std::vector<unsigned> hidden_layers;
	std::vector<std::string> hidden_active;
	parseVectorAuto("HIDDEN_LAYER",hidden_layers,nhidden);
	parseVectorAuto("HIDDEN_ACTIVE",hidden_active,nhidden);
	
	parse("CLIP_LEFT",clip_left);
	parse("CLIP_RIGHT",clip_right);
	
	parseFlag("OPTIMIZE_CONSTANT_PARAMETER",opt_const);
	
	nbiases=numberOfCoeffsSets();
	tot_basis=0;
	for(unsigned i=0;i!=nbiases;++i)
	{
		unsigned nset=Coeffs(i).getSize();
		basises_nums.push_back(nset);
		if(!opt_const)
			--nset;
		tot_basis+=nset;
	}
	
	std::vector<float> init_coe;
	std::string fullname_bias,fullname_wgan;
	
	if(lr_bias.size()==0)
		train_bias=new_traniner(algorithm_bias,pc_bias,fullname_bias);
	else
	{
		if(algorithm_bias=="CyclicalSGD"||algorithm_bias=="cyclicalSGD"||algorithm_bias=="cyclicalsgd"||algorithm_bias=="CSGD"||algorithm_bias=="csgd")
			plumed_massert(lr_bias.size()==2,"The CyclicalSGD algorithm need two learning rates");
		else
			plumed_massert(lr_bias.size()==1,"The "+algorithm_bias+" algorithm need only one learning rates");
		hyper_params_bias.insert(hyper_params_bias.end(),lr_bias.begin(),lr_bias.end());
		
		if(other_params_bias.size()>0)
			hyper_params_bias.insert(hyper_params_bias.end(),other_params_bias.begin(),other_params_bias.end());
		train_bias=new_traniner(algorithm_bias,pc_bias,hyper_params_bias,fullname_bias);
	}

	if(lr_wgan.size()==0)
		train_wgan=new_traniner(algorithm_wgan,pc_wgan,fullname_wgan);
	else
	{
		if(algorithm_wgan=="CyclicalSGD"||algorithm_wgan=="cyclicalSGD"||algorithm_wgan=="cyclicalsgd"||algorithm_wgan=="CSGD"||algorithm_wgan=="csgd")
			plumed_massert(lr_wgan.size()==2,"The CyclicalSGD algorithm need two learning rates");
		else
			plumed_massert(lr_wgan.size()==1,"The "+algorithm_wgan+" algorithm need only one learning rates");
		hyper_params_wgan.insert(hyper_params_wgan.end(),lr_wgan.begin(),lr_wgan.end());
		
		if(other_params_wgan.size()>0)
			hyper_params_wgan.insert(hyper_params_wgan.end(),other_params_wgan.begin(),other_params_wgan.end());
		train_wgan=new_traniner(algorithm_wgan,pc_wgan,hyper_params_wgan,fullname_wgan);
	}
	
	clip_threshold_bias=train_bias->clip_threshold;
	clip_threshold_wgan=train_wgan->clip_threshold;
	
	parse("CLIP_THRESHOLD_BIAS",clip_threshold_bias);
	parse("CLIP_THRESHOLD_WGAN",clip_threshold_wgan);
	
	train_bias->clip_threshold = clip_threshold_bias;
	train_wgan->clip_threshold = clip_threshold_wgan;
	
	unsigned input_dim=narg;
	for(unsigned i=0;i!=nhidden;++i)
	{
		nn_wgan.append(pc_wgan,Layer(input_dim,hidden_layers[i],activation_function(hidden_active[i]),0));
		input_dim=hidden_layers[i];
	}
	nn_wgan.append(pc_wgan,Layer(input_dim,1,LINEAR,0));
	
	parm_bias=pc_bias.add_parameters({1,tot_basis});
	
	init_coe.resize(tot_basis);
	
	if(useMultipleWalkers())
	{
		if(multi_sim_comm.Get_rank()==0)
		{
			if(comm.Get_rank()==0)
				init_coe=as_vector(*parm_bias.values());
			comm.Barrier();
			comm.Bcast(init_coe,0);
		}
		multi_sim_comm.Barrier();
		multi_sim_comm.Bcast(init_coe,0);
	}
	else if(comm.Get_rank()==0)
		init_coe=as_vector(*parm_bias.values());
	comm.Barrier();
	comm.Bcast(init_coe,0);

	unsigned id=0;
	for(unsigned i=0;i!=nbiases;++i)
	{
		std::vector<double> coe(basises_nums[i]);
		for(unsigned j=0;j!=basises_nums[i];++j)
		{		
			if(j==0&&(!opt_const))
				coe[j]=0;
			else
				coe[j]=init_coe[id++];
		}
		Coeffs(i).setValues(coe);
	}
	
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
	
	parse("DEBUG_FILE",debug_file);
	if(debug_file.size()>0)
	{
		is_debug=true;
		odebug.link(*this);
		odebug.open(debug_file);
		odebug.addConstantField("ITERATION");
	}

	std::vector<std::string> arg_perd_str;
	parseVectorAuto("ARG_PERIODIC",arg_perd_str,narg);
	for(unsigned i=0;i!=narg;++i)
	{
		if(arg_perd_str[i]=="Yes"||arg_perd_str[i]=="yes"||arg_perd_str[i]=="YES")
			args_periodic[i]=true;
		else if(arg_perd_str[i]=="No"||arg_perd_str[i]=="no"||arg_perd_str[i]=="NO")
			args_periodic[i]=false;
		else
			plumed_merror("Cannot understand the ARG_PERIODIC type: "+arg_perd_str[i]);
	}
	
	parse("EPOCH_NUM",nepoch);
	plumed_massert(nepoch>0,"EPOCH_NUM must be larger than 0!");
	batch_size=update_steps/nepoch;
	plumed_massert((update_steps-batch_size*nepoch)==0,"UPDATE_STEPS must be divided exactly by EPOCH_NUM");
	
	parseVectorAuto("GRID_MIN",grid_min,narg);
	parseVectorAuto("GRID_MAX",grid_max,narg);
	parseVectorAuto("GRID_BINS",grid_bins,narg);

	ntarget=1;
	for(unsigned i=0;i!=narg;++i)
	{
		grid_space[i]=(grid_max[i]-grid_min[i])/grid_bins[i];
		if(!args_periodic[i])
			++grid_bins[i];
		ntarget*=grid_bins[i];
	}
	
	std::vector<std::vector<float>> target_points;
	std::vector<unsigned> argid(narg,0);
	float p0=1.0/ntarget;
	for(unsigned i=0;i!=ntarget;++i)
	{
		std::vector<float> vtt;
		for(unsigned j=0;j!=narg;++j)
		{
			float tt=grid_min[j]+grid_space[j]*argid[j];
			if(args_periodic[j])
				tt+=grid_space[j]/2.0;
			vtt.push_back(tt);
			input_target.push_back(tt);
			if(j==0)
				++argid[j];
			else if(argid[j-1]==grid_bins[j-1])
			{
				++argid[j];
				argid[j-1]%=grid_bins[j-1];
			}
		}
		target_points.push_back(vtt);
		p_target.push_back(p0);
	}
	
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
			nw=multi_sim_comm.Get_size();
		nepoch*=nw;
		comm.Bcast(nepoch,0);
	}
	
	addComponent("wloss"); componentIsNotPeriodic("wloss");
	valueLwgan=getPntrToComponent("wloss");
	addComponent("bloss"); componentIsNotPeriodic("bloss");
	valueLbias=getPntrToComponent("bloss");
	
	turnOffHessian();
	checkRead();
	
	log.printf("  with random seed: %s\n",std::to_string(random_seed).c_str());
	log.printf("  with lower boundary of the grid:");
	for(unsigned i=0;i!=grid_min.size();++i)
		log.printf(" %f",grid_min[i]);
	log.printf("\n");
	log.printf("  with upper boundary of the grid:");
	for(unsigned i=0;i!=grid_max.size();++i)
		log.printf(" %f",grid_max[i]);
	log.printf("\n");
	log.printf("  with grid bins:");
	for(unsigned i=0;i!=grid_bins.size();++i)
		log.printf(" %d",grid_bins[i]);
	log.printf("\n");
	if(is_debug)
	{
		log.printf("  with target distribution:\n");
		for(unsigned i=0;i!=ntarget;++i)
		{
			log.printf("    target %d:",int(i));
			for(unsigned j=0;j!=narg;++j)
				log.printf(" %f,",target_points[i][j]);
			log.printf(" with %f\n",p_target[i]);
		}
		log.printf("  with total target segments: %d\n",int(ntarget));
	}
	
	log.printf("  with simulation temperature: %f\n",sim_temp);
	log.printf("  with boltzmann constant: %f\n",kB);
	log.printf("  with beta (1/kT): %f\n",beta);
	log.printf("  with bias functions: %d\n",int(nbiases));
	log.printf("  with total optimized coefficients: %d\n",int(tot_basis));
	if(is_debug)
	{
		id=0;
		for(unsigned i=0;i!=nbiases;++i)
		{
			float coe;
			for(unsigned j=0;j!=basises_nums[i];++j)
			{		
				if(j==0&&(!opt_const))
					coe=0;
				else
					coe=init_coe[id++];
				log.printf("    Bias %d with %d initial coefficient: %e and %e\n",int(i),int(j),coe,Coeffs(i).getValue(j));
			}
		}
	}
	
	log.printf("  Use %s to train the W-GAN\n",fullname_wgan.c_str());
	if(lr_wgan.size()>0)
	{
		log.printf("    with learning rates:");
		for(unsigned i=0;i!=lr_wgan.size();++i)
			log.printf(" %f",lr_wgan[i]);
		log.printf("\n");
		if(other_params_wgan.size()>0)
		{
			log.printf("    with other hyperparameters:");
			for(unsigned i=0;i!=other_params_wgan.size();++i)
				log.printf(" %f",other_params_wgan[i]);
			log.printf("\n");
		}
	}
	else
		log.printf("    with default hyperparameters\n");
	log.printf("    with clip threshold: %f\n",clip_threshold_wgan);
	log.printf("    with hidden layers: %d\n",int(nhidden));
	for(unsigned i=0;i!=nhidden;++i)
		log.printf("      Hidden layer %d with dimension %d and activation function \"%s\"\n",int(i),int(hidden_layers[i]),hidden_active[i].c_str());
	log.printf("    with clip range %f to %f\n",clip_left,clip_right);
	
	log.printf("  Use %s to train the neural network of bias function\n",fullname_bias.c_str());
	if(lr_bias.size()>0)
	{
		log.printf("    with learning rates:");
		for(unsigned i=0;i!=lr_bias.size();++i)
			log.printf(" %f",lr_bias[i]);
		log.printf("\n");
		if(other_params_bias.size()>0)
		{
			log.printf("    with other hyperparameters:");
			for(unsigned i=0;i!=other_params_bias.size();++i)
				log.printf(" %f",other_params_bias[i]);
			log.printf("\n");
		}
	}
	else
		log.printf("    with default hyperparameters\n");
	log.printf("    with clip threshold: %f\n",clip_threshold_bias);
	
	log.printf("  Use epoch number %d and batch size %d to update the two networks\n",int(nepoch),int(batch_size));
	log << plumed.cite("Zhang and Noe");
	log.printf("\n");
}

void Opt_TALOS::update()
{
	for(unsigned i=0;i!=narg;++i)
	{
		input_arg.push_back(getArgument(i));
		if(is_debug)
		{
			odebug.printField("time",getTime());
			std::string ff="ARG"+std::to_string(i);
			odebug.printField(ff,input_arg.back());
		}
	}
	
	std::vector<float> debug_vec;
	for(unsigned i=0;i!=nbiases;++i)
	{
		std::vector<double> bsvalues;
		getBiasPntrs()[i]->getBasisSetValues(bsvalues);
		
		unsigned begid=1;
		if(opt_const)
			begid=0;
		
		for(unsigned j=begid;j!=bsvalues.size();++j)
		{
			input_bias.push_back(bsvalues[j]);
			if(is_debug)
				debug_vec.push_back(bsvalues[j]);
		}
	}
	
	if(is_debug)
	{
		std::vector<float> pred(tot_basis,0);
		std::vector<float> ww(tot_basis,0);
		std::vector<float> xx(tot_basis,0);
		if(comm.Get_rank()==0)
		{
			ComputationGraph cg;
			Expression W = parameter(cg,parm_bias);
			Expression x = input(cg,{tot_basis},&debug_vec);
			Expression y_pred = W * x;
			
			if(counts==0)
			{
				ww=as_vector(W.value());
				xx=as_vector(x.value());
			}
			pred=as_vector(cg.forward(y_pred));
		}
		comm.Barrier();
		comm.Bcast(pred,0);
		if(counts==0)
		{
			comm.Bcast(ww,0);
			comm.Bcast(xx,0);
		}
		
		for(unsigned i=0;i!=pred.size();++i)
		{
			std::string ff="PRED"+std::to_string(i);
			odebug.printField(ff,pred[i]);
		}
		odebug.printField();
		odebug.flush();
		
		if(counts==0)
		{
			odebug.printField("time",getTime());
			for(unsigned i=0;i!=ww.size();++i)
			{
				std::string ff="W"+std::to_string(i);
				odebug.printField(ff,ww[i]);
			}
			odebug.printField();
			
			odebug.printField("time",getTime());
			for(unsigned i=0;i!=xx.size();++i)
			{
				std::string ff="x"+std::to_string(i);
				odebug.printField(ff,xx[i]);
			}
			odebug.printField();
			odebug.flush();
		}
	}

	++counts;
	++step;
	
	if(step%update_steps==0&&counts>0)
	{
		comm.Barrier();
		if(input_arg.size()!=narg*update_steps)
			plumed_merror("ERROR! The size of the input_arg mismatch: "+std::to_string(input_arg.size()));
		if(input_bias.size()!=tot_basis*update_steps)
			plumed_merror("ERROR! The size of the input_bias mismatch: "+std::to_string(input_bias.size()));

		double wloss=0;
		double bloss=0;
		std::vector<float> new_coe(tot_basis,0);
		std::vector<std::vector<float>> vec_fw;
		if(useMultipleWalkers())
		{
			multi_sim_comm.Sum(counts);
			
			std::vector<float> all_input_arg;
			std::vector<float> all_input_bias;
			
			all_input_arg.resize(nw*narg*update_steps,0);
			all_input_bias.resize(nw*tot_basis*update_steps,0);
			
			if(comm.Get_rank()==0)
			{
				multi_sim_comm.Allgather(input_arg,all_input_arg);
				multi_sim_comm.Allgather(input_bias,all_input_bias);
			}
			comm.Bcast(all_input_arg,0);
			comm.Bcast(all_input_bias,0);
			
			if(multi_sim_comm.Get_rank()==0)
			{
				if(comm.Get_rank()==0)
				{
					wloss=update_wgan(all_input_arg,vec_fw);
					bloss=update_bias(all_input_bias,vec_fw);
					vec_fw.resize(0);
					new_coe=as_vector(*parm_bias.values());
				}
				comm.Barrier();
				comm.Bcast(wloss,0);
				comm.Bcast(bloss,0);
				comm.Bcast(new_coe,0);
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(wloss,0);
			multi_sim_comm.Bcast(bloss,0);
			multi_sim_comm.Bcast(new_coe,0);
		}
		else
		{
			if(comm.Get_rank()==0)
			{
				wloss=update_wgan(input_arg,vec_fw);
				bloss=update_bias(input_bias,vec_fw);
				vec_fw.resize(0);
				new_coe=as_vector(*parm_bias.values());
			}
		}
		comm.Barrier();
		comm.Bcast(wloss,0);
		comm.Bcast(bloss,0);
		comm.Bcast(new_coe,0);

		valueLwgan->set(wloss);
		valueLbias->set(bloss);
		
		input_arg.resize(0);
		input_bias.resize(0);
		counts=0;
		
		++iter_time;
		
		if(is_debug)
		{
			odebug.printField("ITERATION",int(iter_time));
			odebug.printField("time",getTime());
			for(unsigned i=0;i!=new_coe.size();++i)
			{
				odebug.printField("Coe"+std::to_string(i),new_coe[i]);
			}
			odebug.printField();
			odebug.flush();
		}

		unsigned id=0;
		for(unsigned i=0;i!=nbiases;++i)
		{
			std::vector<double> coe(basises_nums[i]);
			for(unsigned j=0;j!=basises_nums[i];++j)
			{		
				if(j==0&&(!opt_const))
					coe[j]=0;
				else
					coe[j]=new_coe[id++];
			}
			Coeffs(i).setValues(coe);

			unsigned int curr_iter = getIterationCounter()+1;
			double curr_time = getTime();
			getCoeffsPntrs()[i]->setIterationCounterAndTime(curr_iter,curr_time);
		}

		increaseIterationCounter();
		updateOutputComponents();
		for(unsigned int i=0; i<numberOfCoeffsSets(); i++) {
			writeOutputFiles(i);
		}
		if(TartgetDistStride()>0 && getIterationCounter()%TartgetDistStride()==0) {
			for(unsigned int i=0; i<numberOfBiases(); i++) {
				if(DynamicTargetDists()[i]) {
					getBiasPntrs()[i]->updateTargetDistributions();
				}
			}
		}
		if(StrideReweightFactor()>0 && getIterationCounter()%StrideReweightFactor()==0) {
			for(unsigned int i=0; i<numberOfBiases(); i++) {
				getBiasPntrs()[i]->updateReweightFactor();
			}
		}
		
		if(isBiasOutputActive() && getIterationCounter()%getBiasOutputStride()==0) {
			writeBiasOutputFiles();
		}
		if(isFesOutputActive() && getIterationCounter()%getFesOutputStride()==0) {
			writeFesOutputFiles();
		}
		if(isFesProjOutputActive() && getIterationCounter()%getFesProjOutputStride()==0) {
			writeFesProjOutputFiles();
		}
		if(isTargetDistOutputActive() && getIterationCounter()%getTargetDistOutputStride()==0) {
			writeTargetDistOutputFiles();
		}
		if(isTargetDistProjOutputActive() && getIterationCounter()%getTargetDistProjOutputStride()==0) {
			writeTargetDistProjOutputFiles();
		}
	}
}

// training the parameter of WGAN
float Opt_TALOS::update_wgan(const std::vector<float>& all_input_arg,std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Dim xs_dim({narg},batch_size);
	Dim xt_dim({narg},ntarget);
	
	//~ std::random_shuffle(input_target.begin(), input_target.end());

	unsigned wsize=batch_size*narg;
	std::vector<float> input_sample(wsize);
	Expression x_sample=input(cg,xs_dim,&input_sample);
	Expression x_target=input(cg,xt_dim,&input_target);

	Expression y_sample=nn_wgan.run(x_sample,cg);
	Expression y_target=nn_wgan.run(x_target,cg);

	Expression loss_sample=mean_batches(y_sample);
	Expression loss_target=mean_batches(y_target);

	Expression loss_wgan=loss_sample-loss_target;
	
	double wloss=0;
	double loss;
	std::vector<std::vector<float>> vec_input_sample;
	for(unsigned i=0;i!=nepoch;++i)
	{
		for(unsigned j=0;j!=wsize;++j)
			input_sample[j]=all_input_arg[i*wsize+j];
		vec_input_sample.push_back(input_sample);
		loss = as_scalar(cg.forward(loss_wgan));
		wloss += loss;
		cg.backward(loss_wgan);
		train_wgan->update();
		//~ std::cout<<"Loss = "<<loss<<std::endl;
		nn_wgan.clip(clip_left,clip_right);
	}
	wloss/=nepoch;
	
	for(unsigned i=0;i!=nepoch;++i)
	{
		input_sample=vec_input_sample[i];
		vec_fw.push_back(as_vector(cg.forward(y_sample)));
	}
	
	return wloss;
}

// update the coeffients of the basis function
float Opt_TALOS::update_bias(const std::vector<float>& all_input_bias,const std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Expression W = parameter(cg,parm_bias);
	Dim bias_dim({tot_basis},batch_size),fw_dim({1},batch_size);
	unsigned bsize=batch_size*tot_basis;
	std::vector<float> basis_batch(bsize);
	Expression x = input(cg,bias_dim,&basis_batch);
	Expression y_pred = W * x;
	std::vector<float> fw_batch;
	Expression fw = input(cg,fw_dim,&fw_batch);
	
	Expression loss_bias = beta * fw * y_pred;
	Expression loss_mean = mean_batches(loss_bias);

	double bloss=0;
	for(unsigned i=0;i!=nepoch;++i)
	{
		fw_batch=vec_fw[i];
		for(unsigned j=0;j!=bsize;++j)
			basis_batch[j]=all_input_bias[i*bsize+j];
		bloss += as_scalar(cg.forward(loss_mean));
		cg.backward(loss_mean);
		train_bias->update();
	}
	bloss/=nepoch;
	
	//~ new_coe=as_vector(W.value());
	return bloss;
}

Trainer* Opt_TALOS::new_traniner(const std::string& algorithm,ParameterCollection& pc,std::string& fullname)
{
	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		Trainer *trainer = new SimpleSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		Trainer *trainer = new CyclicalSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		Trainer *trainer = new MomentumSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		Trainer *trainer = new AdagradTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		Trainer *trainer = new AdadeltaTrainer(pc);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		Trainer *trainer = new RMSPropTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		Trainer *trainer = new AdamTrainer(pc);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		Trainer *trainer = new AmsgradTrainer(pc);
		return trainer;
	}
	return NULL;
}

Trainer* Opt_TALOS::new_traniner(const std::string& algorithm,ParameterCollection& pc,const std::vector<float>& params,std::string& fullname)
{
	if(params.size()==0)
		return new_traniner(algorithm,pc,fullname);

	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		Trainer *trainer = new SimpleSGDTrainer(pc,params[0]);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		Trainer *trainer=NULL;
		if(params.size()<2)
			plumed_merror("CyclicalSGD needs at least two learning rates");
		else if(params.size()==2)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2]);
		else if(params.size()==4)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3]);
		else
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3],params[4]);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new MomentumSGDTrainer(pc,params[0]);
		else
			trainer = new MomentumSGDTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdagradTrainer(pc,params[0]);
		else
			trainer = new AdagradTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdadeltaTrainer(pc,params[0]);
		else
			trainer = new AdadeltaTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new RMSPropTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new RMSPropTrainer(pc,params[0],params[1]);
		else
			trainer = new RMSPropTrainer(pc,params[0],params[1],params[2]);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdamTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new AdamTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new AdamTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new AdamTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AmsgradTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new AmsgradTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new AmsgradTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new AmsgradTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	return NULL;
}

template<class T>
bool Opt_TALOS::parseVectorAuto(const std::string& keyword, std::vector<T>& values, unsigned num)
{
	plumed_massert(num>0,"the adjust number must be larger than 0!");
	values.resize(0);
	parseVector(keyword,values);
	if(values.size()!=num)
	{
		if(values.size()==1)
		{
			for(unsigned i=1;i!=num;++i)
				values.push_back(values[0]);
		}
		else
			plumed_merror("The number of "+keyword+" must be equal to the number of arguments!");
	}
	return true;
}


}
}

#endif
