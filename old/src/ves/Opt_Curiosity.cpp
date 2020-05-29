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

#include <random>
#include "Optimizer.h"
#include "CoeffsVector.h"
#include "CoeffsMatrix.h"

#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/File.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace ves {

using namespace dytools;

//+PLUMEDOC VES_OPTIMIZER OPT_TALOS
/*
Curiosity sampling

An optimization algorithm for VES using Wasserstein ganerative adversarial network (W-GAN)

\par Algorithm

\par Examples

\endplumedfile

\endplumedfile

*/
//+ENDPLUMEDOC

class Opt_Curiosity :
	public Optimizer,
	public ActionWithArguments
{
private:
	bool is_debug;
	bool opt_const;
	bool opt_rc;
	
	unsigned random_seed;
	
	unsigned narg;
	unsigned nn_input_dim;
	unsigned nbiases;
	unsigned tot_basis;
	unsigned nepoch;
	unsigned batch_size;
	unsigned step;
	unsigned counts;
	unsigned update_steps;
	unsigned nw;
	unsigned iter_time;
	unsigned nn_exp_stride;
	unsigned nrcarg;
	unsigned nn_output_dim;
	unsigned num_nn_fix;
	unsigned num_exp_parm;
	unsigned num_fix_parm;
	
	unsigned tot_data_size;
	unsigned tot_basis_size;
	unsigned tot_in_size;
	unsigned tot_out_size;
	
	const long double PI=3.14159265358979323846264338327950288419716939937510582;
	
	double kB;
	double kBT;
	double beta;
	double sim_temp;
	
	double clip_threshold_bias;
	double clip_threshold_curi;
	
	float clip_left;
	float clip_right;
	
	std::vector<std::string> arg_names;
	
	std::vector<unsigned> basises_nums;
	std::vector<unsigned> const_id;

	std::vector<float> input_rc;
	std::vector<float> input_bias;
	
	std::vector<float> ll1;
	std::vector<float> ul1;
	std::vector<float> ll2;
	std::vector<float> ul2;
	std::vector<float> center1;
	std::vector<float> center2;
	std::vector<float> tsp;
	std::vector<float> state_boundary;
	std::vector<float> ps_boundary;
	
	std::vector<dynet::real> basis_values;
	
	std::string algorithm_bias;
	std::string algorithm_curi;
	
	std::string nn_exp_file;
	std::string nn_fix_file;
	std::string debug_file;
	
	OFile odebug;
	
	std::vector<float> lr_bias;
	std::vector<float> lr_curi;
	std::vector<float> hyper_params_bias;
	std::vector<float> hyper_params_curi;
	
	ParameterCollection pc_bias;
	ParameterCollection pc_exp;
	ParameterCollection pc_fix;
	
	Trainer *train_bias;
	Trainer *train_curi;
	
	Parameter parm_bias;
	MLP nn_exp;
	std::vector<MLP> nn_fix;
	
	Value* valueSwLoss;
	Value* valueVbLoss;
	
	float update_curiosity(const std::vector<float>&,std::vector<std::vector<float>>&);
	std::vector<std::vector<float>> get_fixed_value(const std::vector<float>&);
	float update_bias(const std::vector<float>&,const std::vector<std::vector<float>>&);

public:
	static void registerKeywords(Keywords&);
	explicit Opt_Curiosity(const ActionOptions&);
	~Opt_Curiosity();
	void update();
	void coeffsUpdate(const unsigned int c_id = 0){}
	template<class T>
	bool parseVectorAuto(const std::string&, std::vector<T>&,unsigned);
	template<class T>
	bool parseVectorAuto(const std::string&, std::vector<T>&,unsigned,const T&);
};

PLUMED_REGISTER_ACTION(Opt_Curiosity,"OPT_CURIOSITY")

void Opt_Curiosity::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);
  Optimizer::useFixedStepSizeKeywords(keys);
  Optimizer::useMultipleWalkersKeywords(keys);
  Optimizer::useHessianKeywords(keys);
  Optimizer::useMaskKeywords(keys);
  Optimizer::useRestartKeywords(keys);
  Optimizer::useDynamicTargetDistributionKeywords(keys);
  ActionWithArguments::registerKeywords(keys);
  keys.addOutputComponent("SwLoss","default","loss function of the curiosity/surprisal");
  keys.addOutputComponent("VbLoss","default","loss function of the bias function");
  keys.use("ARG");
  //~ keys.remove("ARG");
  //~ keys.add("compulsory","ARG","the arguments used to set the target distribution");
  keys.remove("STRIDE");
  keys.remove("STEPSIZE");
  keys.add("compulsory","ALGORITHM_BIAS","ADAM","the algorithm to train the neural network of bias function");
  keys.add("compulsory","ALGORITHM_CURI","ADAM","the algorithm to train the curiosity");
  keys.add("compulsory","UPDATE_STEPS","250","the number of step to update");
  keys.add("compulsory","EPOCH_NUM","1","number of epoch for each update per walker");
  keys.add("compulsory","NN_OUTPUT_DIM","1","the ouput dimenision of the curiosity neural network");
  keys.add("compulsory","EXP_HIDDEN_NUMBER","3","the number of hidden layers for exploratory neural network");
  keys.add("compulsory","EXP_HIDDEN_LAYER","32","the dimensions of each hidden layer for exploratory neural network");
  keys.add("compulsory","EXP_HIDDEN_ACTIVE","RELU","active function of each hidden layer for exploratory neural network");
  keys.add("compulsory","FIX_NN_NUMBER","3","the number of hidden layers for fixed neural network");
  keys.add("compulsory","FIX_HIDDEN_NUMBER","3","the number of hidden layers for fixed neural network");
  keys.add("compulsory","FIX_HIDDEN_LAYER","32","the dimensions of each hidden layer fixed neural network");
  keys.add("compulsory","FIX_HIDDEN_ACTIVE","RELU","active function of each hidden layer for fixed neural network");
  keys.add("compulsory","CLIP_LEFT","-0.01","the left value to clip");
  keys.add("compulsory","CLIP_RIGHT","0.01","the right value to clip");
  keys.add("compulsory","NN_EXP_FILE","nn_exp.data","file name of the coefficients of exploratory neural network");
  keys.add("compulsory","NN_EXP_STRIDE","1","the frequency (how many period of update) to out the coefficients of exploratory neural network");
  keys.add("compulsory","NN_FIX_FILE","nn_fix.data","file name of the coefficients of discriminator");
  keys.addFlag("OPT_RC",false,"optimize reaction coordinate during the iteration");
  keys.add("optional","ARG_PERIODIC","if the arguments are periodic or not");
  keys.addFlag("OPT_CONST_PARAM",false,"also to optimize the constant part of the basis functions");
  keys.add("optional","LEARN_RATE_BIAS","the learning rate for training the neural network of bias function");
  keys.add("optional","LEARN_RATE_CURI","the learning rate for training the curiosity");
  keys.add("optional","HYPER_PARAMS_BIAS","other hyperparameters for training the neural network of bias function");
  keys.add("optional","HYPER_PARAMS_CURI","other hyperparameters for training the curiosity");
  keys.add("optional","CLIP_THRESHOLD_BIAS","the clip threshold for training the neural network of bias function");
  keys.add("optional","CLIP_THRESHOLD_CURI","the clip threshold for training the curiosity/surprisal");
  keys.add("optional","SIM_TEMP","the simulation temperature");
  keys.add("optional","DEBUG_FILE","the file to debug");
}

Opt_Curiosity::~Opt_Curiosity() {
	delete train_curi;
	delete train_bias;
	if(is_debug)
		odebug.close();
}

Opt_Curiosity::Opt_Curiosity(const ActionOptions&ao):
  PLUMED_VES_OPTIMIZER_INIT(ao),
  ActionWithArguments(ao),
  is_debug(false),
  random_seed(0),
  narg(getNumberOfArguments()),
  step(0),
  counts(0),
  nw(1),
  iter_time(0),
  train_bias(NULL),
  train_curi(NULL)
{
	random_seed=0;
	if(useMultipleWalkers())
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
	
	int cc=1;
	char pp[]="plumed";
	char *vv[]={pp};
	char** ivv=vv;
	DynetParams params = extract_dynet_params(cc,ivv,true);

	params.random_seed=random_seed;
	
	dynet::initialize(params);
	
	for(unsigned i=0;i!=narg;++i)
	{
		arg_names.push_back(getPntrToArgument(i)->getName());
		//~ log.printf("  %d with argument names: %s\n",int(i),arg_names[i].c_str());
	}

	setStride(1);
	parse("UPDATE_STEPS",update_steps);
	
	parse("ALGORITHM_BIAS",algorithm_bias);
	parse("ALGORITHM_CURI",algorithm_curi);
	
	parseVector("LEARN_RATE_BIAS",lr_bias);
	parseVector("LEARN_RATE_CURI",lr_curi);

	std::vector<float> other_params_bias,other_params_curi;
	parseVector("HYPER_PARAMS_BIAS",other_params_bias);
	parseVector("HYPER_PARAMS_CURI",other_params_curi);

	unsigned nexphidden=0;
	parse("EXP_HIDDEN_NUMBER",nexphidden);
	std::vector<unsigned> exp_hidden_layers;
	std::vector<std::string> exp_hidden_active;
	parseVectorAuto("EXP_HIDDEN_LAYER",exp_hidden_layers,nexphidden);
	parseVectorAuto("EXP_HIDDEN_ACTIVE",exp_hidden_active,nexphidden);
	
	parse("FIX_NN_NUMBER",num_nn_fix);
	unsigned nfixhidden=0;
	parse("FIX_HIDDEN_NUMBER",nfixhidden);
	std::vector<unsigned> fix_hidden_layers;
	std::vector<std::string> fix_hidden_active;
	parseVectorAuto("FIX_HIDDEN_LAYER",fix_hidden_layers,nfixhidden);
	parseVectorAuto("FIX_HIDDEN_ACTIVE",fix_hidden_active,nfixhidden);

	parse("CLIP_LEFT",clip_left);
	parse("CLIP_RIGHT",clip_right);

	parse("NN_FIX_FILE",nn_fix_file);
	parse("NN_EXP_FILE",nn_exp_file);
	parse("NN_EXP_STRIDE",nn_exp_stride);
	
	parseFlag("OPT_CONST_PARAM",opt_const);
	
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
	std::string fullname_bias,fullname_curi;
	
	if(lr_bias.size()==0)
	{
		train_bias=new_traniner(algorithm_bias,pc_bias,fullname_bias);
	}
	else
	{
		if(algorithm_bias=="CyclicalSGD"||algorithm_bias=="cyclicalSGD"||algorithm_bias=="cyclicalsgd"||algorithm_bias=="CSGD"||algorithm_bias=="csgd")
		{
			plumed_massert(lr_bias.size()==2,"The CyclicalSGD algorithm need two learning rates");
		}
		else
		{
			plumed_massert(lr_bias.size()==1,"The "+algorithm_bias+" algorithm need only one learning rates");
		}
		
		hyper_params_bias.insert(hyper_params_bias.end(),lr_bias.begin(),lr_bias.end());
		
		if(other_params_bias.size()>0)
			hyper_params_bias.insert(hyper_params_bias.end(),other_params_bias.begin(),other_params_bias.end());

		train_bias=new_traniner(algorithm_bias,pc_bias,hyper_params_bias,fullname_bias);
	}

	if(lr_curi.size()==0)
	{
		train_curi=new_traniner(algorithm_curi,pc_exp,fullname_curi);
	}
	else
	{
		if(algorithm_curi=="CyclicalSGD"||algorithm_curi=="cyclicalSGD"||algorithm_curi=="cyclicalsgd"||algorithm_curi=="CSGD"||algorithm_curi=="csgd")
		{
			plumed_massert(lr_curi.size()==2,"The CyclicalSGD algorithm need two learning rates");
		}
		else
		{
			plumed_massert(lr_curi.size()==1,"The "+algorithm_curi+" algorithm need only one learning rates");
		}
		
		hyper_params_curi.insert(hyper_params_curi.end(),lr_curi.begin(),lr_curi.end());
		
		if(other_params_curi.size()>0)
			hyper_params_curi.insert(hyper_params_curi.end(),other_params_curi.begin(),other_params_curi.end());
		
		train_curi=new_traniner(algorithm_curi,pc_exp,hyper_params_curi,fullname_curi);
	}

	clip_threshold_bias=train_bias->clip_threshold;
	clip_threshold_curi=train_curi->clip_threshold;

	parse("CLIP_THRESHOLD_BIAS",clip_threshold_bias);
	parse("CLIP_THRESHOLD_CURI",clip_threshold_curi);
	
	train_bias->clip_threshold = clip_threshold_bias;
	train_curi->clip_threshold = clip_threshold_curi;
	
	parseFlag("OPT_RC",opt_rc);
	
	parse("NN_OUTPUT_DIM",nn_output_dim);
	
	nn_input_dim=narg;

	unsigned ldim=nn_input_dim;
	for(unsigned i=0;i!=nexphidden;++i)
	{
		nn_exp.append(pc_exp,Layer(ldim,exp_hidden_layers[i],activation_function(exp_hidden_active[i]),0));
		ldim=exp_hidden_layers[i];
	}
	nn_exp.append(pc_exp,Layer(ldim,nn_output_dim,LINEAR,0));
	num_exp_parm=nn_exp.parameters_number();
	
	for(unsigned i=0;i!=num_nn_fix;++i)
	{
		nn_fix.push_back(MLP());
		ldim=nn_input_dim;
		for(unsigned j=0;j!=nfixhidden;++j)
		{
			nn_fix.back().append(pc_fix,Layer(ldim,fix_hidden_layers[j],activation_function(fix_hidden_active[j]),0));
			ldim=fix_hidden_layers[i];
		}
		nn_fix.back().append(pc_fix,Layer(ldim,nn_output_dim,LINEAR,0));
	}
	num_fix_parm=nn_fix.back().parameters_number();
	
	parm_bias=pc_bias.add_parameters({1,tot_basis});
	
	init_coe.resize(tot_basis);
	
	std::vector<float> params_exp(num_exp_parm);
	std::vector<float> params_fix(num_fix_parm);
	std::vector<float> all_params_fix(num_fix_parm*num_nn_fix);
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				init_coe=as_vector(*parm_bias.values());
				params_exp=nn_exp.get_parameters();
				for(unsigned i=0;i!=num_nn_fix;++i)
				{
					params_fix=nn_fix[i].get_parameters();
					for(unsigned j=0;j!=num_fix_parm;++j)
						all_params_fix[i*num_fix_parm+j]=params_fix[j];
				}
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(init_coe,0);
			multi_sim_comm.Bcast(params_exp,0);
			multi_sim_comm.Bcast(all_params_fix,0);
		}
		comm.Bcast(init_coe,0);
		comm.Bcast(params_exp,0);
		comm.Bcast(all_params_fix,0);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			init_coe=as_vector(*parm_bias.values());
			params_exp=nn_exp.get_parameters();
			for(unsigned i=0;i!=num_nn_fix;++i)
			{
				params_fix=nn_fix[i].get_parameters();
				for(unsigned j=0;j!=num_fix_parm;++j)
					all_params_fix[i*num_fix_parm+j]=params_fix[j];
			}
		}
		comm.Barrier();
		comm.Bcast(init_coe,0);
		comm.Bcast(params_exp,0);
		comm.Bcast(all_params_fix,0);
	}
		
	parm_bias.set_value(init_coe);
	nn_exp.set_parameters(params_exp);
	for(unsigned i=0;i!=num_nn_fix;++i)
	{
		for(unsigned j=0;j!=num_fix_parm;++j)
			params_fix[j]=all_params_fix[i*num_fix_parm+j];
		nn_fix[i].set_parameters(params_fix);
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
		odebug.fmtField(" %f");
		//~ odebug.addConstantField("ITERATION");
	}
	
	parse("EPOCH_NUM",nepoch);
	plumed_massert(nepoch>0,"EPOCH_NUM must be larger than 0!");
	batch_size=update_steps/nepoch;
	plumed_massert((update_steps-batch_size*nepoch)==0,"UPDATE_STEPS must be divided exactly by EPOCH_NUM");
	
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
			nw=multi_sim_comm.Get_size();
		comm.Bcast(nw,0);
		nepoch*=nw;
	}
	
	tot_data_size=update_steps*nw;
	tot_basis_size=tot_data_size*tot_basis;
	tot_in_size=tot_data_size*nn_input_dim;
	tot_out_size=tot_data_size*nn_output_dim;
	
	TextFileSaver saver(nn_fix_file);
	saver.save(pc_fix);
	
	addComponent("SwLoss"); componentIsNotPeriodic("SwLoss");
	valueSwLoss=getPntrToComponent("SwLoss");
	addComponent("VbLoss"); componentIsNotPeriodic("VbLoss");
	valueVbLoss=getPntrToComponent("VbLoss");
	
	turnOffHessian();
	checkRead();
	
	log.printf("  Curiosity Sampling%s\n");
	log.printf("  with random seed: %s\n",std::to_string(random_seed).c_str());
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

	log.printf("  Use %s to train the curiosity\n",fullname_curi.c_str());
	if(lr_curi.size()>0)
	{
		log.printf("    with learning rates:");
		for(unsigned i=0;i!=lr_curi.size();++i)
			log.printf(" %f",lr_curi[i]);
		log.printf("\n");
		if(other_params_curi.size()>0)
		{
			log.printf("    with other hyperparameters:");
			for(unsigned i=0;i!=other_params_curi.size();++i)
				log.printf(" %f",other_params_curi[i]);
			log.printf("\n");
		}
	}
	else
		log.printf("    with default hyperparameters\n");
	log.printf("    with clip threshold: %f\n",clip_threshold_curi);
	log.printf("    with hidden layers of exploratory neural network: %d\n",int(nexphidden));
	for(unsigned i=0;i!=nexphidden;++i)
		log.printf("      Hidden layer %d with dimension %d and activation function \"%s\"\n",int(i),int(exp_hidden_layers[i]),exp_hidden_active[i].c_str());
	log.printf("    with hidden layers of fixed neural network: %d\n",int(nexphidden));
	for(unsigned i=0;i!=nfixhidden;++i)
		log.printf("      Hidden layer %d with dimension %d and activation function \"%s\"\n",int(i),int(fix_hidden_layers[i]),fix_hidden_active[i].c_str());
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
	log << plumed.cite("Zhang, Yang and Noe");
	log.printf("\n");
}

void Opt_Curiosity::update()
{
	std::vector<float> varg;
	if(is_debug)
		odebug.printField("time",getTime());
	for(unsigned i=0;i!=narg;++i)
	{
		float val=getArgument(i);
		input_rc.push_back(val);
		
		if(is_debug)
		{
			std::string ff="ARG"+std::to_string(i);
			odebug.printField(ff,val);
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
			//~ if(is_debug)
				//~ debug_vec.push_back(bsvalues[j]);
		}
	}

	if(is_debug)
	{
		//~ std::vector<float> pred(tot_basis,0);
		//~ std::vector<float> ww(tot_basis,0);
		//~ std::vector<float> xx(tot_basis,0);

		//~ if(comm.Get_rank()==0)
		//~ {
			//~ ComputationGraph cg;
			//~ Expression W = parameter(cg,parm_bias);
			//~ Expression x = input(cg,{tot_basis},&debug_vec);
			//~ Expression y_pred = W * x;
			
			//~ if(counts==0)
			//~ {
				//~ ww=as_vector(W.value());
				//~ xx=as_vector(x.value());
			//~ }
			//~ pred=as_vector(cg.forward(y_pred));
		//~ }
		//~ comm.Barrier();
		//~ comm.Bcast(pred,0);
		//~ if(counts==0)
		//~ {
			//~ comm.Bcast(ww,0);
			//~ comm.Bcast(xx,0);
		//~ }

		//~ for(unsigned i=0;i!=pred.size();++i)
		//~ {
			//~ std::string ff="PRED"+std::to_string(i);
			//~ odebug.printField(ff,pred[i]);
		//~ }

		odebug.printField();
		odebug.flush();

		//~ if(counts==0)
		//~ {
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=ww.size();++i)
			//~ {
				//~ std::string ff="W"+std::to_string(i);
				//~ odebug.printField(ff,ww[i]);
			//~ }
			//~ odebug.printField();
			//~ 
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=xx.size();++i)
			//~ {
				//~ std::string ff="x"+std::to_string(i);
				//~ odebug.printField(ff,xx[i]);
			//~ }
			//~ odebug.printField();
			//~ odebug.flush();
		//~ }
	}

	++counts;
	++step;

	if(step%update_steps==0&&counts>0)
	{
		if(input_rc.size()!=nn_input_dim*update_steps)
			plumed_merror("ERROR! The size of the input_rc mismatch: "+std::to_string(input_rc.size()));
		if(input_bias.size()!=tot_basis*update_steps)
			plumed_merror("ERROR! The size of the input_bias mismatch: "+std::to_string(input_bias.size()));

		double swloss=0;
		double vbloss=0;
		std::vector<float> new_coe(tot_basis,0);
		std::vector<std::vector<float>> vec_Sw;
		std::vector<float> params_exp(num_exp_parm,0);
		
		if(useMultipleWalkers())
		{
			if(comm.Get_rank()==0)
				multi_sim_comm.Sum(counts);
			comm.Bcast(counts,0);

			std::vector<float> all_input_rc;
			std::vector<float> all_input_bias;
			
			all_input_rc.resize(tot_in_size,0);
			all_input_bias.resize(tot_basis_size,0);
			
			if(comm.Get_rank()==0)
			{
				multi_sim_comm.Allgather(input_rc,all_input_rc);
				multi_sim_comm.Allgather(input_bias,all_input_bias);
			}
			comm.Bcast(all_input_rc,0);
			comm.Bcast(all_input_bias,0);

			if(comm.Get_rank()==0)
			{
				if(multi_sim_comm.Get_rank()==0)
				{
					swloss=update_curiosity(all_input_rc,vec_Sw);
					vbloss=update_bias(all_input_bias,vec_Sw);
					vec_Sw.resize(0);
					new_coe=as_vector(*parm_bias.values());
					params_exp=nn_exp.get_parameters();
				}
				multi_sim_comm.Barrier();
				multi_sim_comm.Bcast(swloss,0);
				multi_sim_comm.Bcast(vbloss,0);
				multi_sim_comm.Bcast(new_coe,0);
				multi_sim_comm.Bcast(params_exp,0);
			}
			comm.Barrier();
			comm.Bcast(swloss,0);
			comm.Bcast(vbloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_exp,0);
		}
		else
		{
			if(comm.Get_rank()==0)
			{
				swloss=update_curiosity(input_rc,vec_Sw);
				vbloss=update_bias(input_bias,vec_Sw);
				vec_Sw.resize(0);
				new_coe=as_vector(*parm_bias.values());
				params_exp=nn_exp.get_parameters();
			}
			comm.Barrier();
			comm.Bcast(swloss,0);
			comm.Bcast(vbloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_exp,0);
		}
		
		parm_bias.set_value(new_coe);
		nn_exp.set_parameters(params_exp);

		valueSwLoss->set(swloss);
		valueVbLoss->set(vbloss);
		
		input_rc.resize(0);
		input_bias.resize(0);
		counts=0;
		
		if(comm.Get_rank()==0&&multi_sim_comm.Get_rank()==0&&iter_time%nn_exp_stride==0)
		{
			TextFileSaver saver(nn_exp_file);
			saver.save(pc_exp);
		}
		
		++iter_time;
		
		//~ if(is_debug)
		//~ {
			//~ odebug.printField("ITERATION",int(iter_time));
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=new_coe.size();++i)
			//~ {
				//~ odebug.printField("Coe"+std::to_string(i),new_coe[i]);
			//~ }
			
			//~ odebug.printField();
			//~ odebug.flush();
		//~ }

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
		
		//
		if(isBiasOutputActive() && getIterationCounter()%getBiasOutputStride()==0) {
			writeBiasOutputFiles();
		}
		if(isFesOutputActive() && getIterationCounter()%getFesOutputStride()==0) {
			writeFesOutputFiles();
		}
		if(isFesProjOutputActive() && getIterationCounter()%getFesProjOutputStride()==0) {
			writeFesProjOutputFiles();
		}
	}
}

// training the parameter of discriminator
float Opt_Curiosity::update_curiosity(const std::vector<float>& all_input_rc,std::vector<std::vector<float>>& vec_Sw)
{
	std::vector<std::vector<float>> vec_nn_fix=get_fixed_value(all_input_rc);
	
	ComputationGraph cg;
	Dim input_dim({nn_input_dim},batch_size);
	Dim output_dim({nn_output_dim},batch_size);

	//~ std::random_shuffle(input_target.begin(), input_target.end());

	unsigned in_size=batch_size*nn_input_dim;
	unsigned out_size=batch_size*nn_output_dim;
	std::vector<float> input_explore(in_size);
	std::vector<float> output_fixed(out_size);

	Expression x_explore=input(cg,input_dim,&input_explore);
	Expression y_explore=nn_exp.run(x_explore,cg);
	Expression y_fixed=input(cg,output_dim,&output_fixed);
	
	Expression sw=l2_norm(y_explore-y_fixed);
	Expression loss_sw=sum_batches(sw);
	
	double swloss=0;
	double loss;
	std::vector<std::vector<float>> vec_input_explore;
	for(unsigned i=0;i!=nepoch;++i)
	{
		for(unsigned j=0;j!=in_size;++j)
			input_explore[j]=all_input_rc[i*in_size+j];
		vec_input_explore.push_back(input_explore);
		loss = as_scalar(cg.forward(loss_sw));
		swloss += loss;
		cg.backward(loss_sw);
		train_curi->update();
		//~ nn_curi.clip(clip_left,clip_right);
	}
	swloss/=nepoch;
	
	for(unsigned i=0;i!=nepoch;++i)
	{
		input_explore=vec_input_explore[i];
		vec_Sw.push_back(as_vector(cg.forward(sw)));
	}
	return swloss;
}

std::vector<std::vector<float>> Opt_Curiosity::get_fixed_value(const std::vector<float>& all_input_rc)
{
	unsigned in_size=batch_size*nn_input_dim;
	unsigned out_size=batch_size*nn_output_dim;
	std::vector<float> input_sample(in_size);
	std::vector<std::vector<float>> input_sample_batches;

	for(unsigned i=0;i!=nepoch;++i)
	{
		for(unsigned j=0;j!=in_size;++j)
			input_sample[j]=all_input_rc[i*in_size+j];
		input_sample_batches.push_back(input_sample);
	}

	Dim xs_dim({nn_input_dim},batch_size);
	std::vector<std::vector<float>> vec_nn_fix(nepoch,std::vector<float>(out_size,0));
	for(unsigned i=0;i!=num_nn_fix;++i)
	{
		ComputationGraph cg;		
		Expression x=input(cg,xs_dim,&input_sample);
		Expression y=nn_fix[i].run(x,cg);

		for(unsigned j=0;j!=nepoch;++j)
		{
			input_sample=input_sample_batches[j];
			std::vector<float> vec_tmp=as_vector(cg.forward(y));
			for(unsigned k=0;k!=out_size;++k)
				vec_nn_fix[j][k]+=vec_tmp[k];
		}
	}
	return vec_nn_fix;
}

// update the coeffients of the basis function
float Opt_Curiosity::update_bias(const std::vector<float>& all_input_bias,const std::vector<std::vector<float>>& vec_Sw)
{
	float avg_sw=0;
	for(unsigned i=0;i!=vec_Sw.size();++i)
	{
		for(unsigned j=0;j!=vec_Sw[i].size();++j)
			avg_sw+=vec_Sw[i][j];
	}
	avg_sw/=tot_data_size;
	
	ComputationGraph cg;
	Expression W = parameter(cg,parm_bias);
	Dim bias_dim({tot_basis},batch_size),sw_dim({1},batch_size);
	unsigned bsize=batch_size*tot_basis;
	std::vector<float> basis_batch(bsize);
	Expression x = input(cg,bias_dim,&basis_batch);
	Expression y_pred = W * x;
	std::vector<float> sw_batch;
	Expression sw = input(cg,sw_dim,&sw_batch);
	
	Expression loss_mean1 = mean_batches(sw * y_pred);
	Expression loss_mean2 = mean_batches(avg_sw * y_pred);
	
	Expression loss_fin = beta * (loss_mean1 - loss_mean2);
	
	double vbloss=0;
	for(unsigned i=0;i!=nepoch;++i)
	{
		sw_batch=vec_Sw[i];
		for(unsigned j=0;j!=bsize;++j)
			basis_batch[j]=all_input_bias[i*bsize+j];
		vbloss += as_scalar(cg.forward(loss_fin));
		cg.backward(loss_fin);
		train_bias->update();
	}
	vbloss/=nepoch;
	
	//~ new_coe=as_vector(W.value());
	return vbloss;
}

template<class T>
bool Opt_Curiosity::parseVectorAuto(const std::string& keyword, std::vector<T>& values, unsigned num)
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

template<class T>
bool Opt_Curiosity::parseVectorAuto(const std::string& keyword, std::vector<T>& values, unsigned num,const T& def_value)
{
	plumed_massert(num>0,"the adjust number must be larger than 0!");
	values.resize(0);
	parseVector(keyword,values);
	
	if(values.size()!=num)
	{
		if(values.size()==0)
		{
			for(unsigned i=0;i!=num;++i)
				values.push_back(def_value);
		}
		else if(values.size()==1)
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
