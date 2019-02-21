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
#include "ITS_Bias.h"

namespace PLMD{
namespace bias{

//+PLUMEDOC BIAS ITS_BIAS
/*
Used to perform integrated tempering sampling (ITS) molecular dynamics
simulation on potential energy of the system or bias potential of CVs.

The ITS method generates a broad distribution as a function of the
potential energy \f$U\f$ of the system, which is achieved by using an
effective potential energy \f$U^{\text{eff}}\f$ :

\f[
U^{\text{eff}} = -\frac{1}{\beta_0} \log \sum_{k}^{N} n_k e^{- \beta_k U}
\f]

in which \f$\beta_0\f$ is the temperature of the system of interest,
\f$\beta_k\f$ is a series of temperatures that cover both low and high
temperatures, and \f$\{n_k\}\f$ are reweighting factors obtained through
an iterative procedure.

In PLUMED, all the bias are CV (collective variables) based. Therefore,
we change the form of ITS method to a CV-based bias potential \f$V(U)\f$:

\f[
V(U) = U^{\text{eff}} - U = -\frac{1}{\beta_0} \log \sum_{k}^{N} n_k e^{- \beta_k U} - U
\f]

In this form, the CV should be only chosen as energy, for example, the
potential energy of the system. In fact, the bias energy from the PLUMED
can be also used as the energy, in this cituation:

\f[
V(V') = -\frac{1}{\beta_0} \log \sum_{k}^{N} n_k e^{- \beta_0 (a_k V')} - V'
\f]

Therefore, the ITS method can be also consider about a broad distribution
as a summation of a series of rescaled bias potential. \f$a_k\f$ is the
rescale factor. In fact, \f$a_k\f$ and \f$\beta_k\f$ are equivalent because
\f$\beta_k = a_k * \beta_0\f$ then \f$a_k = T_0/T_k\f$. Therefore, you can choose the series of
temperatures or rescale factors to generate the broad distribution.

In order to obtain the weight factors \f$\{n_k\}\f$ at different temperature,
we use a iteration process during the ITS simulation. The principle of
setting the weight factors \f$\{n_k\}\f$ is let the distribution of potential
energy at different temperatures have the same contribution:

\f[
P_i=n_i \int_{r} e^{- \beta_i U} dr = n_j \int_{r} e^{- \beta_j U} dr = P_j
\f]
 
The system can be recovered to the thermodynamics at normal temperature
\f$\beta_0\f$ by multiplying a reweighting factor \f$c_0\f$ on the
probabilities of each observation:

\f[
c_0 = e^{\beta_0(U - U^{\text{eff}})} =
\left[\sum_{k}^{N} n_k e^{(\beta_0 - \beta_k)U}\right]^{-1}
\f]

\par Examples

\verbatim
energy: ENERGY

ITS_BIAS ...
  LABEL=its
  ARG=energy
  NREPLICA=100
  SIM_TEMP=300
  TEMP_MAX=370
  TEMP_MIN=270
  PACE=2000
  PESHIFT=1500
  FB_FILE=fb.data
  FB_STRIDE=100
  FBTRAJ_FILE=fbtrj.data
  FBTRAJ_STRIDE=20
... ITS_BIAS
\endverbatim

\verbatim
energy: ENERGY

DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
METAD ARG=d1,d2 SIGMA=0.2,0.2 HEIGHT=0.3 PACE=500 LABEL=restraint

ITS_BIAS ...
  LABEL=its
  ARG=energy
  BIAS=restraint
  NREPLICA=100
  SIM_TEMP=300
  TEMP_MAX=370
  TEMP_MIN=270
  PACE=2000
  PESHIFT=1500
  FB_FILE=fb.data
  FB_STRIDE=100
  FBTRAJ_FILE=fbtrj.data
  FBTRAJ_STRIDE=20
... ITS_BIAS
\endverbatim

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(ITS_Bias,"ITS_BIAS")

void ITS_Bias::registerKeywords(Keywords& keys)
{
	Bias::registerKeywords(keys);
	keys.addOutputComponent("rbias","default","the revised bias potential using rct");
	keys.addOutputComponent("rct","default","the reweighting revise factor");
	keys.addOutputComponent("energy","default","the instantaneous value of the potential energy of the system");
	keys.addOutputComponent("Ueff","default","the instantaneous value of the effective potential");
	keys.addOutputComponent("Teff","default","the instantaneous value of the bias force");
#ifdef __PLUMED_HAS_DYNET
	keys.addOutputComponent("wloss","USE_TALITS","loss function of the W-GAN");
	keys.addOutputComponent("bloss","USE_TALITS","loss function of the neural network of bias function");
#endif
	keys.addOutputComponent("rwfb","DEBUG_FILE","the revised bias potential using rct");
	ActionWithValue::useCustomisableComponents(keys);
	keys.remove("ARG");
    keys.add("optional","ARG","the argument(CVs) here must be the potential energy. If no argument is setup, only bias will be integrated in this method."); 
	//~ keys.use("ARG");

	keys.add("compulsory","NREPLICA","the number of the replicas");
	keys.add("compulsory","PESHIFT","0.0","the shift value of potential energy");
	keys.add("compulsory","PACE","1000","the frequency for updating fb value");
	
	keys.add("optional","BIAS","the label of the bias to be used in the bias-ITS method");
	keys.add("optional","BIAS_RATIO","the ratio of the bias to be used in the bias-ITS method");
	keys.add("optional","SIM_TEMP","the temperature used in the simulation");
	keys.add("optional","TEMP_MIN","the lower bounds of the temperatures");
	keys.add("optional","TEMP_MAX","the upper bounds of the temperatures");
	keys.add("optional","RATIO_MIN","the minimal ratio of temperature");
	keys.add("optional","RATIO_MAX","the maximal ratio of temperature");
	keys.add("optional","TARGET_TEMP","the temperature that the system distribution at");
	
	keys.addFlag("EQUIVALENT_TEMPERATURE",false,"to simulate the system at target temperatue but keep using the original thermal bath");
	keys.addFlag("FB_FIXED",false,"to fix the fb value without update");
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
	keys.addFlag("VARIATIONAL",false,"use variational approach to iterate the fb value");
	keys.addFlag("TEMP_CONTRIBUTE",false,"use the contribution of each temperatue to calculate the derivatives instead of the target distribution");
	keys.addFlag("PESHIFT_AUTO_ADJUST",false,"automatically adjust the PESHIFT value during the fb iteration");
	keys.addFlag("UNLINEAR_REPLICAS",false,"to setup the segments of temperature be propotional to the temperatues. If you setup the REPLICA_RATIO_MIN value, this term will be automatically opened.");
	keys.addFlag("DIRECT_AVERAGE",false,"to directly calculate the average of rbfb value in each step (only be used in traditional iteration process)");
	keys.addFlag("NOT_USE_BIAS_RCT",false,"do not use the c(t) of bias to modify the bias energy when using bias as the input CVs");
	
#ifdef __PLUMED_HAS_DYNET
	keys.addFlag("USE_TALITS",false,"use targeted adversarial learning ITS");
	keys.add("optional","ALGORITHM_BIAS","the algorithm to train the neural network of bias function");
	keys.add("optional","ALGORITHM_WGAN","the algorithm to train the W-GAN");
	keys.add("optional","TARGET_MIN","the lower bounds of the target distribution of effective temperatures");
	keys.add("optional","TARGET_MAX","the upper bounds of the target distribution of effective temperatures");
	keys.add("optional","TARGET_BINS","the number of bins of the target distribution of effective temperatures");
	keys.add("optional","TARGETDIST_FILE","read target distribution from file");
	keys.add("optional","EPOCH_NUM","number of epoch for each update per walker");
	keys.add("optional","PRE_UPDATE","");
	keys.add("optional","HIDDEN_NUMBER","the number of hidden layers for W-GAN");
	keys.add("optional","HIDDEN_LAYER","the dimensions of each hidden layer  for W-GAN");
	keys.add("optional","HIDDEN_ACTIVE","active function of each hidden layer  for W-GAN");
	keys.add("optional","CLIP_LEFT","the left value to clip");
	keys.add("optional","CLIP_RIGHT","the right value to clip");
	keys.add("optional","WGAN_FILE","file name of the coefficients of W-GAN");
	keys.add("optional","WGAN_OUTPUT","the frequency (how many period of update) to out the coefficients of W-GAN");
	keys.add("optional","LEARN_RATE_BIAS","the learning rate for training the neural network of bias function");
	keys.add("optional","LEARN_RATE_WGAN","the learning rate for training the W-GAN");
	keys.add("optional","HYPER_PARAMS_BIAS","other hyperparameters for training the neural network of bias function");
	keys.add("optional","HYPER_PARAMS_WGAN","other hyperparameters for training the W-GAn");
	keys.add("optional","CLIP_THRESHOLD_BIAS","the clip threshold for training the neural network of bias function");
	keys.add("optional","CLIP_THRESHOLD_WGAN","the clip threshold for training the W-GAn");
#endif

	keys.add("optional","START_CYCLE","the start step for fb updating");
	keys.add("optional","FB_INIT","( default=0.0 ) the default value for fb initializing");
	keys.add("optional","RB_FAC1","( default=0.5 ) the ratio of the average value of rb");
	keys.add("optional","RB_FAC2","( default=0.0 ) the ratio of the old steps in rb updating");
	keys.add("optional","STEP_SIZE","( default=1.0 )the step size of fb iteration");
	keys.add("optional","TARGET_RATIO_ENERGY","( default=0 ) the energy to adjust the ratio of each temperatures during the iteration");
	keys.add("optional","RCT_FILE","the file to output the c(t)");
	keys.add("optional","RCT_STRIDE","the frequency to output the c(t)");

	keys.add("optional","FB_FILE","( default=fb.data ) a file to record the new fb values when they are update");
	keys.add("optional","FB_STRIDE","( default=1 ) the frequency to output the new fb values");
	keys.add("optional","FBTRAJ_STRIDE","( default=FB_STRIDE )the frequency to record the evoluation of fb values");
	keys.add("optional","FBTRAJ_FILE"," a file recording the evolution of fb values");
	//~ keys.add("optional","NORM_TRAJ"," a file recording the evolution of normalize factors");
	//~ keys.add("optional","ITER_TRAJ"," a file recording the evolution of the fb iteration factors");
	//~ keys.add("optional","DERIV_TRAJ"," a file recording the evolution of the derivation of fb factors");
	//~ keys.add("optional","PESHIFT_TRAJ"," a file recording the evolution of peshift");
	
	keys.add("optional","RW_TEMP","the temperatures used in the calcaulation of reweighting factors");
	//~ keys.addFlag("REVISED_REWEIGHT",false,"calculate the c(t) value at reweighting factor output file");
	keys.add("optional","FB_READ_FILE","a file of reading fb values (include temperatures and peshift)");
	keys.add("optional","BIAS_FILE","a file to output the function of bias potential");
	keys.add("optional","RBFB_FILE","a file to output the evoluation of rbfb");
	keys.add("optional","BIAS_STRIDE","the frequency to output the bias potential");
	keys.add("optional","BIAS_MIN","the minimal value of coordinate of the bias potential function");
	keys.add("optional","BIAS_MAX","the maximal value of coordinate of the bias potential function");
	keys.add("optional","BIAS_BIN","the number of bins to record the bias potential function");
	keys.add("optional","DEBUG_FILE","a file of output debug information");
	keys.add("optional","POTDIS_FILE","the file that record the distribution of potential energy during the simulation");
	keys.add("optional","POTDIS_STEP","the freuency to record the distribution of potential energy");
	keys.add("optional","POTDIS_UPDATE","the freuency to output the distribution of potential energy");
	keys.add("optional","POTDIS_MIN","the minimal value of potential in the potential distribution file");
	keys.add("optional","POTDIS_MAX","the maximal value of potential in the potential distribution file");
	keys.add("optional","POTDIS_BIN","the number of bins that record the potential distribution");
	keys.add("optional","ENERGY_MIN","the minimal potential energy to calculate the 2nd order derivative");
	keys.add("optional","ENERGY_MAX","the maximal potential energy to calculate the 2nd order derivative");
	keys.add("optional","ENERGY_ACCURACY","the interval of potential energy to calculate the 2nd order derivative");

	keys.add("optional","ITERATE_LIMIT","to limit the iteration of fb value in variational in order to prevent the weight of higher temperatues become larger than the lower one");
}

ITS_Bias::~ITS_Bias()
{
#ifdef __PLUMED_HAS_DYNET
	if(use_talits)
	{
		delete train_wgan;
		delete train_bias;
	}
#endif
	if(!equiv_temp&&!is_const)
	{
		ofb.close();
		ofbtrj.close();
		if(rct_output)
			orct.close();
		//~ onormtrj.close();
		//~ opstrj.close();

	}
	if(is_debug)
		odebug.close();
	if(rbfb_output)
		orbfb.close();
}

ITS_Bias::ITS_Bias(const ActionOptions& ao):
	PLUMED_BIAS_INIT(ao),update_start(0),ratio_energy(0),rct(0),
	step(0),norm_step(0),mcycle(0),iter_limit(0),
	fb_init(0.0),fb_bias(0.0),rb_fac1(0.5),rb_fac2(0.0),step_size(1.0),
	is_const(false),rw_output(false),
	read_norm(false),bias_output(false),
	rbfb_output(false),is_debug(false),
	bias_linked(false),only_bias(false),is_read_ratio(false),
	is_set_temps(false),is_set_ratios(false),is_norm_rescale(false),
	read_fb(false),read_iter(false),fbtrj_output(false),rct_output(false),
	partition_initial(false),
	start_cycle(0),fb_stride(1),bias_stride(1),rctid(0)
#ifdef __PLUMED_HAS_DYNET
	,train_bias(NULL),train_wgan(NULL),nn_wgan(pc_wgan)
#endif
{
	if(getNumberOfArguments()==0)
		only_bias=true;
	else if(getNumberOfArguments()>1)
		plumed_merror("this edition of ITS can only accept one CV");
		
	std::vector<std::string> bias_labels(0);
	parseVector("BIAS",bias_labels);
	nbiases = bias_labels.size();
	if(nbiases>0)
	{
		bias_linked=true;
		bias_pntrs_.resize(nbiases);
		for(unsigned int i=0; i<nbiases; i++)
		{
			bias_pntrs_[i]=plumed.getActionSet().selectWithLabel<Bias*>(bias_labels[i]);
			if(!bias_pntrs_[i]){plumed_merror("bias "+bias_labels[i]+" does not exist. NOTE: the Bias-ITS should always be defined AFTER all the other biases.");}
			bias_pntrs_[i]->linkExternalBias(this);
		}
		parseVector("BIAS_RATIO",bias_ratio);
		if(bias_ratio.size()==1)
		{
			double tbr=bias_ratio[0];
			bias_ratio.assign(nbiases,tbr);
		}
		else if(bias_ratio.size()!=nbiases)
			bias_ratio.resize(nbiases,1.0);
	}
	else if(only_bias)
		plumed_merror("the quantity of ARG and BIAS must be setup at last one");

	kB=plumed.getAtoms().getKBoltzmann();

	parseFlag("FB_FIXED",is_const);
	parseFlag("EQUIVALENT_TEMPERATURE",equiv_temp);
	parseFlag("MULTIPLE_WALKERS",use_mw);
	
	parse("PACE",update_step);
	if(update_step==0)
		plumed_merror("PACE cannot be 0");
#ifdef __PLUMED_HAS_DYNET
	parseFlag("USE_TALITS",use_talits);
	
	std::vector<float> other_params_bias,other_params_wgan;
	std::vector<unsigned> hidden_layers;
	std::vector<std::string> hidden_active;
	std::string fullname_bias,fullname_wgan;
	if(use_talits&&!is_const&&!equiv_temp)
	{
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
		
		int cc=1;
		char pp[]="plumed";
		char *vv[]={pp};
		char** ivv=vv;
		DynetParams params = extract_dynet_params(cc,ivv,true);

		params.random_seed=random_seed;
		
		dynet::initialize(params);
		
		nepoch=1;
		parse("EPOCH_NUM",nepoch);
		plumed_massert(nepoch>0,"EPOCH_NUM must be larger than 0!");
		batch_size=update_step/nepoch;
		plumed_massert((update_step-batch_size*nepoch)==0,"UPDATE_STEPS must be divided exactly by EPOCH_NUM");
		
		algorithm_bias="ADAM";
		parse("ALGORITHM_BIAS",algorithm_bias);
		algorithm_wgan="ADAM";
		parse("ALGORITHM_WGAN",algorithm_wgan);
		
		parseVector("LEARN_RATE_BIAS",lr_bias);
		parseVector("LEARN_RATE_WGAN",lr_wgan);
		
		parseVector("HYPER_PARAMS_BIAS",other_params_bias);
		parseVector("HYPER_PARAMS_BIAS",other_params_wgan);
		
		unsigned nhidden=3;
		parse("HIDDEN_NUMBER",nhidden);
	
		parseVector("HIDDEN_LAYER",hidden_layers);
		if(hidden_layers.size()!=nhidden)
		{
			if(hidden_layers.size()==0)
				hidden_layers.assign(nhidden,8);
			else if(hidden_layers.size()==1)
			{
				unsigned hl0=hidden_layers[0];
				hidden_layers.assign(nhidden,hl0);
			}
			else
				plumed_merror("The number of HIDDEN_LAYER must be equal to HIDDEN_NUMBER!");
		}
		
		parseVector("HIDDEN_ACTIVE",hidden_active);
		if(hidden_active.size()!=nhidden)
		{
			if(hidden_active.size()==0)
				hidden_active.assign(nhidden,"RELU");
			else if(hidden_active.size()==1)
			{
				std::string ha0=hidden_active[0];
				hidden_active.assign(nhidden,ha0);
			}
			else
				plumed_merror("The number of HIDDEN_ACTIVE must be equal to HIDDEN_NUMBER!");
		}
		
		clip_left=-0.01;
		parse("CLIP_LEFT",clip_left);
		clip_right=0.01;
		parse("CLIP_RIGHT",clip_right);
		
		//~ parse("WGAN_FILE",wgan_file);
		//~ parse("WGAN_OUTPUT",wgan_output);
		
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
		
		unsigned ldim=1;
		for(unsigned i=0;i!=nhidden;++i)
		{
			nn_wgan.append(pc_wgan,Layer(ldim,hidden_layers[i],activation_function(hidden_active[i]),0));
			ldim=hidden_layers[i];
		}
		nn_wgan.append(pc_wgan,Layer(ldim,1,LINEAR,0));
		
		std::vector<float> params_wgan(nn_wgan.parameters_number());
		if(use_mw)
		{
			if(comm.Get_rank()==0)
			{
				if(multi_sim_comm.Get_rank()==0)
					params_wgan=nn_wgan.get_parameters();
				multi_sim_comm.Barrier();
				multi_sim_comm.Bcast(params_wgan,0);
			}
			comm.Bcast(params_wgan,0);
		}
		else
		{
			if(comm.Get_rank()==0)
				params_wgan=nn_wgan.get_parameters();
			comm.Barrier();
			comm.Bcast(params_wgan,0);
		}
		nn_wgan.set_parameters(params_wgan);
	}
#endif

	parseFlag("PESHIFT_AUTO_ADJUST",auto_peshift);
	parseFlag("UNLINEAR_REPLICAS",is_unlinear);
	parseFlag("DIRECT_AVERAGE",is_direct);
	parseFlag("NOT_USE_BIAS_RCT",no_bias_rct);
	parse("PESHIFT",peshift);
	parse("FB_READ_FILE",fb_input);
	fb_file="fb.data";
	parse("FB_FILE",fb_file);
	
	if(only_bias&&equiv_temp)
		plumed_merror("EQUIVALENT_TEMPERATURE must be used with the potential energy as the argument");

	unsigned _nreplica(0);
	double _templ(-1),_temph(-1),_ratiol(-1),_ratioh(-1);
	parse("NREPLICA",_nreplica);
	if(_nreplica==0&&fb_input.size()==0)
		plumed_merror("NREPLICA must be set up or read from file.");
	parse("TEMP_MIN",_templ);
	parse("TEMP_MAX",_temph);
	if(_templ>0&&_temph>0)
		is_set_temps=true;
	parse("RATIO_MIN",_ratiol);
	parse("RATIO_MAX",_ratioh);
	parse("TARGET_RATIO_ENERGY",ratio_energy);
	
	if(_ratiol>0||_ratioh>0)
	{
		if(is_set_temps&&fb_input.size()==0)
			plumed_merror("the range of temperatures (TEMP_MIN and TEMP_MAX) and the range of replica ratio (REPLICA_RATIO_MIN and REPLICA_RATIO_MAX) cannot be setup simultaneously.");
		is_set_ratios=true;
		if(_ratiol<0)
		{
			if(is_unlinear)
				plumed_merror("if you want to use unliner temperatures, the minimal replica ratio (REPLICA_RATIO_MIN) must be setup!");
			_ratiol=0.0;
		}
		else
			is_unlinear=true;
		if(_ratioh<0) _ratioh=1.0;
	}
	if(!is_set_ratios&&!is_set_temps&&fb_input.size()==0)
		plumed_merror("the range of temperatures must be setup (TEMP_MIN/TEMP_MAX or REPLICA_RATIO_MIN/REPLICA_RATIO_MAX) or read from file (FB_READ_FILE).");
	
	parse("START_CYCLE",start_cycle);

	double _kB=kB;
	double _peshift=peshift;
	
	sim_temp=-1;
	parse("SIM_TEMP",sim_temp);
	if(sim_temp>0)
		kBT=kB*sim_temp;
	else
	{
		kBT=plumed.getAtoms().getKbT();
		sim_temp=kBT/kB;
	}
	beta0=1.0/kBT;
	
	parse("START_CYCLE",start_cycle);
	
	unsigned read_count=0;
	
	if(fb_input.size()>0)
	{
		read_fb=true;
		read_count=read_fb_file(fb_input,_kB,_peshift);
		if(read_count==0)
			plumed_merror("do not read anything in the file "+fb_input);
		comm.Barrier();
		if(use_mw && comm.Get_rank()==0)
			multi_sim_comm.Barrier();
	}
	else if(getRestart())
	{
		read_count=read_fb_file(fb_file,_kB,_peshift);
		if(read_count==0)
			plumed_merror("do not read anything in the file "+fb_file);
		comm.Barrier();
		if(use_mw && comm.Get_rank()==0)
			multi_sim_comm.Barrier();
	}
	else
	{
		if(_nreplica<1)
			plumed_merror("the number of temperatures much be larger than 1");
		nreplica=_nreplica+1;

		if(is_set_ratios)
		{
			if(_ratiol<0)
				plumed_merror("REPLICA_RATIO_MIN must be large than 0");
			ratiol=_ratiol;

			if(_ratioh<0)
				plumed_merror("REPLICA_RATIO_MAX must be large than 0");
			ratioh=_ratioh;
			if(ratiol>=ratioh)
				plumed_merror("the value of RATIO_BIAS_MAX must be large than RATIO_BIAS_MIN");
				
			templ=sim_temp*ratiol;
			temph=sim_temp*ratioh;
			
			if(!is_unlinear)
			{
				double dr=(ratioh-ratiol)/_nreplica;
				for(unsigned i=0;i!=nreplica;++i)
				{
					double ratio_now=ratiol+i*dr;
					int_ratios.push_back(ratio_now);
					int_temps.push_back(sim_temp*ratio_now);
				}
			}
		}
		else
		{
			if(_templ<0)
				plumed_merror("TEMP_MIN must be large than 0");
			templ=_templ;

			if(_temph<0)
				plumed_merror("TEMP_MAX must be large than 0");
			temph=_temph;
			if(templ>=temph)
				plumed_merror("the value of TEMP_MAX must be large than TEMP_MIN");
				
			ratiol=templ/sim_temp;
			ratioh=temph/sim_temp;
			
			if(!is_unlinear)
			{
				double dt=(temph-templ)/_nreplica;
				for(unsigned i=0;i!=nreplica;++i)
				{
					double temp_now=templ+i*dt;
					int_temps.push_back(temp_now);
					int_ratios.push_back(temp_now/sim_temp);
				}
			}
		}
		if(is_unlinear)
		{
			double temp_ratio=exp(std::log(temph/templ)/_nreplica);
			for(unsigned i=0;i!=nreplica;++i)
			{
				double temp_now=templ*pow(temp_ratio,i);
				int_temps.push_back(temp_now);
				int_ratios.push_back(temp_now/sim_temp);
			}
		}
		mcycle=start_cycle;
		++start_cycle;
	}
	for(unsigned i=0;i!=nreplica;++i)
		betak.push_back(1.0/(kB*int_temps[i]));

	logN=std::log(double(nreplica));
	ratio_norm=logN;
	fb_ratio0=-logN;
	// the target distribution of P'_k=Z_k/\sum_i Z_i, default value is 1/N
	fb_ratios.assign(nreplica,-logN);
	// the target ratio of the two neighor distribution R_k=P'_{k+1}/P'_{k}, default value is 1:1
	rbfb_ratios.assign(nreplica,0);
	
	if(fabs(ratio_energy)>1.0e-15)
	{
		double ratio1=-ratio_energy*betak[0];
		ratio_norm=ratio1;
		fb_ratios[0]=ratio1;
		for(unsigned i=1;i!=nreplica;++i)
		{
			double ratio_value=-ratio_energy*betak[i];
			fb_ratios[i]=ratio_value;
			exp_added(ratio_norm,ratio_value);
			rbfb_ratios[i-1]=fb_ratios[i]-fb_ratios[i-1];
		}
		fb_ratio0=-ratio_energy*beta0-ratio_norm;
		
		for(unsigned i=0;i!=nreplica;++i)
			fb_ratios[i]-=ratio_norm;
	}

	target_temp=sim_temp;
	parse("TARGET_TEMP",target_temp);
	if(equiv_temp)
	{
		kBT_target=kB*target_temp;
		beta_target=1.0/kBT_target;
		eff_factor=beta_target/beta0;
		bias_force=1-eff_factor;
	}
	
	parse("FB_INIT",fb_init);
	parse("RB_FAC1",rb_fac1);
	parse("RB_FAC2",rb_fac2);
	parse("STEP_SIZE",step_size);
	
	if(fabs(step_size-1)>1.0e-6)
	{
		is_norm_rescale=true;
		fb_bias=std::log(step_size);
	}
	
	if(!read_norm)
		norml.assign(nreplica,0);
	
	gU.assign(nreplica,0);
	gE.assign(nreplica,0);
	gf.assign(nreplica,0);
	bgf.assign(nreplica,0);
	rbfb.assign(nreplica,0);
	rbzb.assign(nreplica,0);
	fb_rct.assign(nreplica,0);
	peshift_ratio.assign(nreplica,0);

	parse("FB_STRIDE",fb_stride);

	parse("FBTRAJ_FILE",fb_trj);
	if(fb_trj.size()>0)
		fbtrj_output=true;
	fbtrj_stride=fb_stride;
	parse("FBTRAJ_STRIDE",fbtrj_stride);
	
	parse("RCT_FILE",rct_file);
	if(rct_file.size()>0)
		rct_output=true;
	rct_stride=fb_stride;
	parse("RCT_STRIDE",rct_stride);

	set_peshift_ratio();
	if(!read_fb&&!getRestart())
	{
		for(unsigned i=0;i!=nreplica;++i)
			fb.push_back(fb_init*(int_temps[i]-int_temps[0]));
	}
	
	rctid=find_rw_id(sim_temp,sim_dtl,sim_dth);
	
#ifdef __PLUMED_HAS_DYNET
	if(use_talits)
	{
		parse("TARGETDIST_FILE",targetdis_file);
		if(targetdis_file.size()>0)
		{
			IFile itarget;
			itarget.link(*this);
			itarget.open(targetdis_file.c_str());
			itarget.allowIgnoredFields();

			if(!itarget.FieldExist("temp"))
				plumed_merror("Cannot found Field \"temp\"");
			if(!itarget.FieldExist("targetdist"))
				plumed_merror("Cannot found Field \"targetdist\"");

			double tmin;
			if(itarget.FieldExist("max_temp"))
				itarget.scanField("max_temp",tmin);
			target_min=tmin;

			double tmax;
			if(itarget.FieldExist("max_temp"))
				itarget.scanField("max_temp",tmax);
			target_max=tmax;
			
			int tbins;
			if(itarget.FieldExist("nbins_temp"))
				itarget.scanField("nbins_temp",tbins);
			else
				plumed_merror("Cannot found Field \"nbins_temp\"");
				
			if(tbins>0)
				ntarget=tbins;
			else
				plumed_merror("nbins_temp must be larger than 0!");
				
			double sum=0;
			for(unsigned i=0;i!=ntarget;++i)
			{
				double tt;
				itarget.scanField("temp",tt);
				temps_target.push_back(tt);
				
				double p0;
				itarget.scanField("targetdist",p0);
				temps_dis.push_back(p0);
				
				sum+=p0;
				itarget.scanField();
			}
			itarget.close();
			comm.Barrier();
			for(unsigned i=0;i!=ntarget;++i)
				temps_dis[i]/=(sum/ntarget);
		}
		else
		{
			target_min=templ;
			parse("TARGET_MIN",target_min);
			target_max=temph;
			parse("TARGET_MAX",target_max);
			ntarget=nreplica-1;
			parse("TARGET_BINS",ntarget);
			
			target_space=(target_max-target_min)/ntarget;
			++ntarget;
			
			for(unsigned i=0;i!=ntarget;++i)
			{
				temps_dis.push_back(1.0);
				temps_target.push_back(target_min+i*target_space);
			}
		}

		parm_bias=pc_bias.add_parameters({nreplica-1});
		std::vector<float> init_coe;
		
		if(!read_fb)
		{
			init_coe.resize(nreplica-1);
			if(use_mw)
			{
				if(comm.Get_rank()==0)
				{
					if(multi_sim_comm.Get_rank()==0)
						init_coe=as_vector(*parm_bias.values());
					multi_sim_comm.Barrier();
					multi_sim_comm.Bcast(init_coe,0);
				}
				comm.Barrier();
				comm.Bcast(init_coe,0);
			}
			else
			{
				if(comm.Get_rank()==0)
					init_coe=as_vector(*parm_bias.values());
				comm.Barrier();
				comm.Bcast(init_coe,0);
			}
		}
		
		float fb1=init_coe[0];
		if(rctid==0)
			fb1=0;
		
		unsigned tmpid=0;
		for(unsigned i=0;i!=nreplica;++i)
		{
			if(i==rctid)
				fb[i]=-fb1;
			else
				fb[i]=init_coe[tmpid++]-fb1;
		}
		
		parm_bias.set_value(init_coe);
	}
#endif

	if(!partition_initial)
		partition.resize(nreplica,-1e38);

	if(!equiv_temp&&!is_const)
	{
		setupOFile(fb_file,ofb,use_mw);
		ofb.addConstantField("ITERATE_METHOD");
		ofb.addConstantField("ITERATE_STEP");
		ofb.addConstantField("BOLTZMANN_CONSTANT");
		ofb.addConstantField("PESHIFT");
		ofb.addConstantField("COEFFICIENT_TYPE");
		ofb.addConstantField("NREPLICA");
		ofb.addConstantField("TARGET_RATIO_ENERGY");

		if(fbtrj_output)
			setupOFile(fb_trj,ofbtrj,use_mw);
	}
	
	parse("BIAS_FILE",bias_file);
	if(bias_file.size()>0)
	{
		bias_output=true;
		bias_max=0;
		parse("BIAS_MAX",bias_max);
		bias_min=0;
		parse("BIAS_MIN",bias_min);
		if(bias_max<=bias_min)
			plumed_merror("BIAS_MAX must be larger than bias_min");
		parse("BIAS_BIN",bias_bins);
		if(bias_bins<=1)
			plumed_merror("BIAS_BIN must be larger than 1");
		parse("BIAS_STRIDE",bias_stride);
		d_pot=(bias_max-bias_min)/(bias_bins-1);
	}
	
	parse("RBFB_FILE",rbfb_file);
	if(rbfb_file.size()>0)
	{
		rbfb_output=true;
		orbfb.link(*this);
		orbfb.open(rbfb_file);
	}

	parse("ITERATE_LIMIT",iter_limit);
	
	fb0=find_rw_fb(rctid,sim_dtl,sim_dth);
	rct=calc_rct(beta0,fb0,fb_ratio0);
	
	addComponent("rbias"); componentIsNotPeriodic("rbias");
	valueRBias=getPntrToComponent("rbias");
	setRctComponent("rct");
	setRct(rct);
	//~ addComponent("force"); componentIsNotPeriodic("force");
	//~ valueForce=getPntrToComponent("force");
	addComponent("energy"); componentIsNotPeriodic("energy");
	valuePot=getPntrToComponent("energy");
	addComponent("Ueff"); componentIsNotPeriodic("Ueff");
	valueUeff=getPntrToComponent("Ueff");
	addComponent("Teff"); componentIsNotPeriodic("Teff");
	valueTeff=getPntrToComponent("Teff");
	
#ifdef __PLUMED_HAS_DYNET
	addComponent("wloss"); componentIsNotPeriodic("wloss");
	valueLwgan=getPntrToComponent("wloss");
	addComponent("bloss"); componentIsNotPeriodic("bloss");
	valueLbias=getPntrToComponent("bloss");
#endif

	parseVector("RW_TEMP",rw_temp);

	std::vector<double> rw_fb(rw_temp.size());
	if(rw_temp.size()>0)
	{
		rw_output=true;

		rw_rct.resize(rw_temp.size());
		rw_rctid.resize(rw_temp.size());
		rw_dth.resize(rw_temp.size());
		rw_dtl.resize(rw_temp.size());
		rw_fb_ratios.resize(rw_temp.size());
		for(unsigned i=0;i!=rw_temp.size();++i)
		{
			if(rw_temp[i]<=temph&&rw_temp[i]>=templ)
				rw_beta.push_back(1.0/(kB*rw_temp[i]));
			else
				plumed_merror("the reweighting temperature must between TEMP_MIN and TEMP_MAX");
			rw_rctid[i]=find_rw_id(rw_temp[i],rw_dtl[i],rw_dth[i]);
			rw_fb[i]=find_rw_fb(rw_rctid[i],rw_dtl[i],rw_dth[i]);
			rw_rct[i]=calc_rct(rw_beta[i],rw_fb[i],fb_ratios[i]);
			rw_fb_ratios[i]=-ratio_energy*rw_beta[i]-ratio_norm;
		}

		rw_factor.resize(rw_temp.size());
		valueRwbias.resize(rw_temp.size());
		for(unsigned i=0;i!=rw_temp.size();++i)
		{
			std::string tt;
			Tools::convert(rw_temp[i],tt);
			std::string rtt="rbias_T"+tt;
			addComponent(rtt); componentIsNotPeriodic(rtt);
			valueRwbias[i]=getPntrToComponent(rtt);
		}
	}

	parse("DEBUG_FILE",debug_file);
	if(debug_file.size()>0)
	{
		is_debug=true;
		odebug.link(*this);
		odebug.open(debug_file);
		
		odebug.addConstantField("ITERATE_STEP");
		odebug.addConstantField("NORM_STEP");
		odebug.addConstantField("STEP_SIZE");
		odebug.addConstantField("PESHIFT");
	}
	
	if(is_debug)
	{
		addComponent("rwfb"); componentIsNotPeriodic("rwfb");
		valueRwfb=getPntrToComponent("rwfb");
		valueRwfb->set(fb0);
	}

	checkRead();

	log.printf("  with simulation temperature: %f\n",sim_temp);
	log.printf("  with boltzmann constant: %f\n",kB);
	log.printf("  wiht beta (1/kT): %f\n",beta0);
	log.printf("  wiht target ratio energy: %f\n",ratio_energy);
	log.printf("  wiht fb ratio at simulation temperature (%f): %f\n",sim_temp, fb_ratio0);
	
	if(equiv_temp)
	{
		log.printf("  with equivalent target temperature: %f\n",target_temp);
		log.printf("  The system will run on target temperature (%fK) but keep the old thermal bath (%f).\n",target_temp,sim_temp);
	}
	else
	{
		if(only_bias)
			log.printf("  Only bias potential are setup as the CVs.\n");
		if(bias_linked)
		{
			if(nbiases==1)
				log.printf("  linked with bias \"%s\", and the ratio of bias is %f.\n",bias_labels[0].c_str(),bias_ratio[0]);
			else
				log.printf("  linked with %d biases:\n",nbiases);
			for(unsigned i=0;i!=nbiases;++i)
				log.printf("    Bias %d: %s with ratio %f\n",i+1,bias_labels[i].c_str(),bias_ratio[i]);
			if(no_bias_rct)
				log.printf("    without using the c(t) of the bias\n");
			else
				log.printf("    using the c(t) of the bias if necessary.\n");
		}
		log.printf("  with basic parameters:\n");
		log.printf("    FB_INIT: %f\n",fb_init);
		log.printf("    FB_BIAS: %f\n",fb_bias);
		log.printf("    RB_FAC1: %f\n",rb_fac1);
		log.printf("    RB_FAC2: %f\n",rb_fac2);
		if(read_fb)
		{
			log.printf("  Reading in FB values from file: %s\n",fb_input.c_str());
			if(read_count==1)
				log.printf("  the reading iteration step is %d\n",int(mcycle));
			else
				log.printf("  with totally reading %d sets of coefficients, and the lastest one of iteration step %d will be used\n",int(read_count),int(mcycle));
		}
		else if(getRestart())
		{
			log.printf("  Restart running with reading in FB values from file: %s\n",fb_file.c_str());
			log.printf("  with totally reading %d sets of coefficients, and the lastest one of iteration step %d will be used\n",int(read_count),int(mcycle));
		}
		else if(is_unlinear)
			log.printf("  Unlinear replicas is open");

		if(read_norm)
		{
			log.printf("  with temperatue and FB value: (index T_k gamma_k beta_k P'_k fb norml)\n");
			for(unsigned i=0;i!=nreplica;++i)
				log.printf("    %d\t%f\t%f\t%f\t%f\t%f\t%f\n",i,int_temps[i],int_ratios[i],betak[i],fb_ratios[i],fb[i],norml[i]);
		}
		else
		{
			log.printf("  with temperatue and FB value: (index[k] T_k gamma_k beta_k P'_k fb_k)\n");
			for(unsigned i=0;i!=nreplica;++i)
				log.printf("    %d\t%f\t%f\t%f\t%f\t%f\n",i,int_temps[i],int_ratios[i],betak[i],fb_ratios[i],fb[i]);
		}
		log.printf("    with temperatues of FB from %f to %f\n",templ,temph);
		log.printf("    with ratio of replica from %f to %f\n",ratiol,ratioh);
		
		log.printf("    with number of replica: %d\n",_nreplica);
		log.printf("    with PESHIFT value: %f\n",peshift);
		log.printf("    using the linear interpolation between the fb values of %d-th (%fK) and %d-th (%fK)\n",int(rctid),int_temps[rctid],int(rctid+1),int_temps[rctid+1]);
		log.printf("    with the parameter %f and %f\n",sim_dtl,sim_dth);
		if(fabs(_kB/kB-1)>1.0e-8)
			log.printf("    with Original PESHIFT value: %f (with boltzmann constant %f)\n",
				_peshift,_kB);
		if(use_mw)
		{
			log.printf("  Using multiple walkers");
			log.printf("   with number of walkers: %d\n",multi_sim_comm.Get_size());
			log.printf("   with walker number: %d\n",multi_sim_comm.Get_rank());
			log.printf("\n");
		}
#ifdef __PLUMED_HAS_DYNET
		if(use_talits)
		{
			log.printf("  Using targeted adversarial learning method to iterate the fb value\n");
			if(targetdis_file.size()>0)
				log.printf("  with target distribution of effective temperatures read from file: %s\n",targetdis_file.c_str());
			for(unsigned i=0;i!=ntarget;++i)
				log.printf("    %d. %f K with %f\n",int(i),temps_target[i],temps_dis[i]);
			log.printf("\n");
		}
#endif
		if(!is_const)
		{
			log.printf("  Using ITS update\n");
			if(is_direct)
				log.printf("    with directly calculating the avaerage of rbfb at each step\n");
			else
				log.printf("    with calculating the avaerage of rbfb at the end of fb updating\n");
			log.printf("    with frequence of FB value update: %d\n",
				update_step);
			log.printf("    writing FB output to file: %s\n",fb_file.c_str());
			log.printf("    writing FB trajectory to file: %s\n",fb_trj.c_str());
			if(rbfb_output)
				log.printf("    writing RBFB trajectory to file: %s\n",rbfb_file.c_str());
			log.printf("    writing normalized factors trajectory to file: %s\n",
				norm_trj.c_str());
			log.printf("    writing potential energy shift trajectory to file: %s\n",
				peshift_trj.c_str());
			log.printf("\n");
		}
		if(rw_output)
		{
			log.printf("    with reweighting factor at temperature:\n");
			for(unsigned i=0;i!=rw_temp.size();++i)
				log.printf("    %d\t%fK(%f) fit at %d-th temperature with target ratio %f\n",int(i),rw_temp[i],rw_beta[i],int(rw_rctid[i]),rw_fb_ratios[i]);
		}
		if(is_debug)
			log.printf("  Using debug mod with output file: %s\n",debug_file.c_str());

		log<<"Bibliography "<<
			plumed.cite("Gao, J. Chem. Phys. 128, 064105 (2008)")<<" "<<
			plumed.cite("Yang, Niu and Parrinello, J. Phys. Chem. Lett. 9, 6426 (2018)");
		log<<"\n";
	}
	if(bias_output)
	{
		log.printf("  Wirting output bias potential function\n");
		log.printf("    with output range from %f to %f\n",bias_min,bias_max);
		log.printf("    with output bins: %d\n",bias_bins);
		log.printf("    with Frequence of output: %d\n",bias_stride);
		log.printf("    wirting bias output to file: %s\n",bias_file.c_str());
		log.printf("\n");
	}
}

void ITS_Bias::calculate()
{
	if(only_bias)
		energy=0;
	else
		energy=getArgument(0);
		
	shift_pot=energy+peshift;

	cv_energy=energy;
	
	if(bias_linked)
	{
		tot_bias=0;
		for(unsigned i=0;i!=nbiases;++i)
		{
			tot_bias += bias_ratio[i] * bias_pntrs_[i]->getBias();
			if((!no_bias_rct)&&bias_pntrs_[i]->isSetRct())
				tot_bias -= bias_pntrs_[i]->getRct();
		}
		cv_energy += tot_bias;
	}
	++step;

	// U = U_total + E_shift
	input_energy=cv_energy+peshift;
	
	double eff_temp=sim_temp;

	if(equiv_temp)
	{
		eff_energy=input_energy*eff_factor;
		bias_energy=-1*input_energy*bias_force;
		eff_temp=sim_temp/eff_factor;
	}
	else
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			// -\beta_k*U_pot
			gU[i]=-betak[i]*shift_pot;
			// -\beta_k*U
			gE[i]=-betak[i]*input_energy;
			// log[n_k*exp(-\beta_k*U)]
			gf[i]=gE[i]+fb[i];
			// log[\beta_k*n_k*exp(-\beta_k*U)]
			bgf[i]=gf[i]+std::log(betak[i]);
		}

		// log{\sum_k[n_k*exp(-\beta_k*U)]}
		gfsum=gf[0];
		// log{\sum_k[\beta_k*n_k*exp(-\beta_k*U)]}
		bgfsum=bgf[0];
		for(unsigned i=1;i!=nreplica;++i)
		{
			//~ exp_added(gUsum,gE[i]);
			exp_added(gfsum,gf[i]);
			exp_added(bgfsum,bgf[i]);
		}

		// U_EFF=-1/\beta_0*log{\sum_k[n_k*exp(-\beta_k*U)]}
		eff_energy=-gfsum/beta0;
		bias_energy=eff_energy-input_energy;
		eff_factor=exp(bgfsum-gfsum)/beta0;
		bias_force=1.0-eff_factor;
		eff_temp=sim_temp/eff_factor;
	}

	setBias(bias_energy);
	valueRBias->set(bias_energy-rct);
	
	if(bias_linked)
	{
		for(unsigned i=0;i!=nbiases;++i)
			bias_pntrs_[i]->setExtraBiasRatio(-1.0*bias_force*bias_ratio[i]);
	}

	if(!only_bias)
		setOutputForce(0,bias_force);

	valuePot->set(shift_pot);
	valueUeff->set(eff_energy);
	valueTeff->set(eff_temp);
	
#ifdef __PLUMED_HAS_DYNET
	if(use_talits)
	{
		temps_sample.push_back(eff_temp);
		//~ temps_sample.push_back(cv_energy);
		energy_sample.push_back(input_energy);
	}
#endif

	if(!equiv_temp)
	{
		if(!is_const)
		{
			if(is_direct)
				update_rbfb_direct();
			else
				update_rbfb();
			++norm_step;
		}
		
		if(cv_energy<min_ener)
			min_ener=cv_energy;

		if(bias_output&&step==0)
			output_bias();

		if(rw_output)
		{
			for(unsigned i=0;i!=rw_factor.size();++i)
			{

				rw_factor[i]=calc_bias(rw_beta[i]);
				valueRwbias[i]->set(rw_factor[i]-rw_rct[i]);
			}
		}

		if(is_debug)
		{
			if(!is_const)
			{
				energy_record.push_back(energy);
				gfsum_record.push_back(gfsum);
				bgfsum_record.push_back(bgfsum);
				effpot_record.push_back(eff_energy);
				bias_record.push_back(bias_energy);
				force_record.push_back(bias_force);
			}
			else
			{
				odebug.printField("step",int(step));
				odebug.printField("energy",energy);
				if(bias_linked)
					odebug.printField("tot_bias",tot_bias);
				odebug.printField("cv_energy",cv_energy);
				odebug.printField("peshift",peshift);
				odebug.printField("input_energy",input_energy);
				odebug.printField("eff_energy",eff_energy);
				odebug.printField("bias_energy",bias_energy);
				odebug.printField();
			}
		}

		if(!is_const && step%update_step==update_start && step>update_start)
		{
			if(is_debug)
			{
				for(unsigned i=0;i!=energy_record.size();++i)
				{
					int rs=step-energy_record.size()+i;
					odebug.printField("step",rs);
					odebug.printField("energy",energy_record[i]);
					odebug.printField("gfsum",gfsum_record[i]);
					odebug.printField("bgfsum",bgfsum_record[i]);
					odebug.printField("effective",effpot_record[i]);
					odebug.printField("bias",bias_record[i]);
					odebug.printField("force",force_record[i]);
					odebug.printField();
				}
				energy_record.resize(0);
				gfsum_record.resize(0);
				bgfsum_record.resize(0);
				effpot_record.resize(0);
				bias_record.resize(0);
				force_record.resize(0);
			}
			
			++mcycle;
			
			if(rbfb_output&&mcycle%fbtrj_stride==0)
			{
				orbfb.fmtField(" %f");
				orbfb.printField("step",int(mcycle));
				for(unsigned i=0;i!=nreplica;++i)
				{
					std::string id;
					Tools::convert(i,id);
					std::string rbfbid="RBFB"+id;
					orbfb.printField(rbfbid,rbfb[i]);
				}
				orbfb.printField();
				orbfb.flush();
			}

			//~ comm.Sum(rbfb);
			if(use_mw)
				mw_merge_rbfb();
			if(auto_peshift&&-1.0*min_ener>peshift)
				change_peshift(-1.0*min_ener);

			fb_iteration();
			
#ifdef __PLUMED_HAS_DYNET
			if(use_talits)
			{
				temps_sample.resize(0);
				energy_sample.resize(0);
			}
#endif
			
			fb0=find_rw_fb(rctid,sim_dtl,sim_dth);
			rct=calc_rct(beta0,fb0,fb_ratio0);
			setRct(rct);
			
			for(unsigned i=0;i!=nreplica;++i)
				fb_rct[i]=calc_rct(betak[i],fb[i]);
			
			if(rw_output)
			{
				for(unsigned i=0;i!=rw_rct.size();++i)
					rw_rct[i]=calc_rct(rw_beta[i],find_rw_fb(rw_rctid[i],rw_dtl[i],rw_dth[i]),rw_fb_ratios[i]);
			}
			
			if(is_debug)
			{
				valueRwfb->set(fb0);
			}
				
			if(is_debug)
				odebug.flush();

			output_fb();
			for(unsigned i=0;i!=nreplica;++i)
				rbfb[i]=-1e38;

			norm_step=0;
		}
	}
}

// The iterate process of rbfb
// rbfb[k]=log[\sum_t(Z_k)]
inline void ITS_Bias::update_rbfb()
{
	// the summation record the data of each the update steps (default=100)
	if(norm_step==0)
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			rbfb[i]=gf[i];
			rbzb[i]=gU[i]-gfsum-betak[i]*fb_rct[i];
		}
	}
	else
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			exp_added(rbfb[i],gf[i]);
			exp_added(rbzb[i],gU[i]-gfsum-betak[i]*fb_rct[i]);
		}
	}
}

// The iterate process of rbfb
// rbfb[k]=log[\sum_t(P_k)]; P_k=Z_k/[\sum_k(Z_k)];
// The equivalence in variational iterate process:
// rbfb[i]=log[(\sum_t(\beta*(\partial V_bias(U;a)/\partial a_i)))_V(a)]
inline void ITS_Bias::update_rbfb_direct()
{
	//~ if(step%update_step==update_start)
	if(norm_step==0)
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			rbfb[i]=gf[i]-gfsum;
			rbzb[i]=gU[i]-gfsum-betak[i]*fb_rct[i];
		}
	}
	else
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			exp_added(rbfb[i],gf[i]-gfsum);
			exp_added(rbzb[i],gU[i]-gfsum-betak[i]*fb_rct[i]);
		}
	}
}

void ITS_Bias::mw_merge_rbfb()
{
	multi_sim_comm.Sum(norm_step);
	
	unsigned nw=0;
	if(comm.Get_rank()==0)
		nw=multi_sim_comm.Get_size();
	comm.Bcast(nw,0);
	
	std::vector<double> all_min_ener(nw,0);
	std::vector<double> all_rbfb(nw*nreplica,0);
	std::vector<double> all_rbzb(nw*nreplica,0);
	if(comm.Get_rank()==0)
	{
		multi_sim_comm.Allgather(rbfb,all_rbfb);
		multi_sim_comm.Allgather(rbzb,all_rbzb);
		multi_sim_comm.Allgather(min_ener,all_min_ener);
	}
	comm.Bcast(all_rbfb,0);
	comm.Bcast(all_rbzb,0);
	comm.Bcast(all_min_ener,0);

	min_ener=all_min_ener[0];
	for(unsigned j=0;j!=rbfb.size();++j)
	{
		rbfb[j]=all_rbfb[j];
		rbzb[j]=all_rbzb[j];
	}
	for(unsigned i=1;i<nw;++i)
	{
		if(all_min_ener[i]<min_ener)
			min_ener=all_min_ener[i];
		for(unsigned j=0;j!=rbfb.size();++j)
		{
			exp_added(rbfb[j],all_rbfb[i*rbfb.size()+j]);
			exp_added(rbzb[j],all_rbzb[i*rbzb.size()+j]);
		}
	}
	if(is_direct)
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			rbfb[i]-=std::log(static_cast<double>(nw));
			rbzb[i]-=std::log(static_cast<double>(nw));
		}
	}
}

// Y. Q. Gao, J. Chem. Phys. 128, 134111 (2008)
void ITS_Bias::fb_iteration()
{
	if(is_direct)
	{
		for(unsigned i=0;i!=nreplica;++i)
		{
			rbfb[i]-=std::log(static_cast<double>(update_step));
			rbzb[i]-=std::log(static_cast<double>(update_step));
		}
	}
	else
	{
		double rbfbsum=rbfb[0];
		for(unsigned i=1;i!=nreplica;++i)
			exp_added(rbfbsum,rbfb[i]);
		for(unsigned i=0;i!=nreplica;++i)
		{
			rbfb[i]-=rbfbsum;
			rbzb[i]-=std::log(static_cast<double>(update_step));
		}
	}

	if(partition_initial)
	{
		for(unsigned i=0;i!=nreplica;++i)
			exp_added(partition[i],rbzb[i]);
	}
	else
	{
		for(unsigned i=0;i!=nreplica;++i)
			partition[i]=rbzb[i];
		partition_initial=true;
	}
	
	// ratio[k]=log[m_k(t)]
	std::vector<double> ratio;
	std::vector<double> old_fb;
	if(is_debug)
		old_fb=fb;
	if(mcycle==start_cycle&&!read_norm)
	{
		for(unsigned i=0;i!=nreplica-1;++i)
		{
			norml[i]=(rbfb[i]+rbfb[i+1])*rb_fac1;
			// m_k(0)=n_k(0)/n_{k+1}(0)
			double ratio_old=fb[i]-fb[i+1];
			// m_k(1)={m'}_k(1)=m_k(0)*P_k(0)/P_{k+1}(0)/R_k(0)
			ratio.push_back(ratio_old+rbfb[i+1]-rbfb[i]-rbfb_ratios[i]);
		}
	}
	else
	{
		for(unsigned i=0;i!=nreplica-1;++i)
		{
			// rb=log[f(P_k,P_{k+1};t)]
			// f(P_k,P_{k+1};t)=(P_k(t)*P_{k+1}(t))^c1+t*c2
			// (default f(P_k,P_{k+1};t)=\sqrt[P_k(t)*P_{k+1}(t)])
			double rb=(rbfb[i]+rbfb[i+1])*rb_fac1+(mcycle-1)*rb_fac2;
			// ratio_old=log[m_k(t-1)], m_k=n_k/n_{k+1}
			// (Notice that m_k in the paper equal to n_{k+1}/n_k)
			double ratio_old=fb[i]-fb[i+1];
			// ratio_new=log[{m'}_k(t)]=log[m_k(t-1)*P_{k+1}/P_{k}//R_k], if P_{k+1}/P_{k}=R_k, m_k(t)=m_k(t-1)
			double ratio_new=ratio_old+rbfb[i+1]-rbfb[i]-rbfb_ratios[i];

			// normal=log[W_k(t)], normalold=log[W_k(t-1)]
			// W_k(t)=\sum_t[f(P_k,P_{k+1})]=W_k(t-1)+f(P_k,P_{k+1};t)
			// the summation record the data of ALL the update steps
			double normlold=norml[i];
			exp_added(norml[i],rb);

			// Rescaled normalization factor
			double rationorm=norml[i];
			// the weight of new ratio
			// w_k(t)=c_bias*f(P_k,P_{k+1};t)
			double weight=fb_bias+rb;
			// Wr_k(t)=W_k(t-1)+w_k(t)
			if(is_norm_rescale)
				rationorm=exp_add(normlold,weight);

			// m_k(t)=[m_k(t-1)*c_bias*f(P_k,P_{k+1};t)*P_k(t)/P_{k+1}(t)+m_k(t-1)*W_k(t-1)]/Wr_k(t)
			// m_k(t)=[{m'}_k(t)*w_k(t)+m_k(t-1)*W_k(t-1)]/Wr_k(t)
			ratio.push_back(exp_add(weight+ratio_new,normlold+ratio_old)-rationorm);
		}
	}

	// partio[k]=log[1/n_k]
	std::vector<double> pratio(nreplica,0);
	// 1/n_k=n_1*\prod_j^k(m_j)
	for(unsigned i=0;i!=nreplica-1;++i)
		pratio[i+1]=pratio[i]+ratio[i];

	// As in the paper m_k=n_{k+1}/n_k
	// So here weights[k]=log[n_k]=log{1/pratio[k]}=-log{pratio[k]}
	
	//~ double rescalefb=logN+fb[rctid]-betak[rctid]*peshift;

	bool is_nan=false;
	for(unsigned i=0;i!=nreplica;++i)
	{
		fb[i]=-pratio[i];
		if(fb[i]!=fb[i])
			is_nan=true;
	}
	if(is_debug||is_nan)
	{
		if(!is_debug)
		{
			odebug.link(*this);
			odebug.open("error.data");
		}

		odebug.printf("--- FB update ---\n");
		for(unsigned i=0;i!=nreplica;++i)
		{
			std::string id;
			Tools::convert(i,id);
			odebug.printField("index",id);
			if(is_set_ratios)
				odebug.printField("ratio",int_ratios[i]);
			else
				odebug.printField("temperature",int_temps[i]);
			odebug.printField("betak",betak[i]);
			odebug.printField("fb",fb[i]);
			odebug.printField("old_fb",old_fb[i]);
			odebug.printField("rbfb",rbfb[i]);
			odebug.printField("rct",fb_rct[i]);
			odebug.printField("rct_rbfb",(fb_rct[i]-old_fb[i])/betak[i]);
			odebug.printField();
		}
		odebug.printf("--- FB update ---\n");
	}
	if(is_nan)
		plumed_merror("FB value become NaN. See debug file or \"error.data\" file for detailed information.");
		
#ifdef __PLUMED_HAS_DYNET
	if(use_talits)
	{
		std::vector<float> new_coe;
		float fb_sim=fb[rctid];
		for(unsigned i=0;i!=nreplica;++i)
		{
			if(i!=rctid)
				new_coe.push_back(fb[i]-fb_sim);
		}
		parm_bias.set_value(new_coe);

		double wloss=0;
		double bloss=0;	
		std::vector<std::vector<float>> vec_fw;
		std::vector<float> params_wgan(nn_wgan.parameters_number(),0);
		if(use_mw)
		{
			unsigned nw=1;
			if(comm.Get_rank()==0)
				nw=multi_sim_comm.Get_size();
			comm.Bcast(nw,0);

			std::vector<float> all_temps_sample;
			std::vector<float> all_energy_sample;
			
			all_temps_sample.resize(nw*update_step,0);
			all_energy_sample.resize(nw*update_step,0);
			
			if(comm.Get_rank()==0)
			{
				multi_sim_comm.Allgather(temps_sample,all_temps_sample);
				multi_sim_comm.Allgather(energy_sample,all_energy_sample);
			}
			comm.Bcast(all_temps_sample,0);
			comm.Bcast(all_energy_sample,0);

			if(comm.Get_rank()==0)
			{
				if(multi_sim_comm.Get_rank()==0)
				{
					wloss=update_wgan(all_temps_sample,vec_fw);
					bloss=update_bias(all_energy_sample,vec_fw);
					vec_fw.resize(0);
					new_coe=as_vector(*parm_bias.values());
					params_wgan=nn_wgan.get_parameters();
				}
				multi_sim_comm.Barrier();
				multi_sim_comm.Bcast(wloss,0);
				multi_sim_comm.Bcast(bloss,0);
				multi_sim_comm.Bcast(new_coe,0);
				multi_sim_comm.Bcast(params_wgan,0);
			}
			comm.Barrier();
			comm.Bcast(wloss,0);
			comm.Bcast(bloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_wgan,0);
		}
		else
		{
			if(comm.Get_rank()==0)
			{
				wloss=update_wgan(temps_sample,vec_fw);
				bloss=update_bias(energy_sample,vec_fw);
				vec_fw.resize(0);
				new_coe=as_vector(*parm_bias.values());
				params_wgan=nn_wgan.get_parameters();
			}
			comm.Barrier();
			comm.Bcast(wloss,0);
			comm.Bcast(bloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_wgan,0);
		}
		parm_bias.set_value(new_coe);
		nn_wgan.set_parameters(params_wgan);
		
		valueLwgan->set(wloss);
		valueLbias->set(bloss);
		
		float fb1=new_coe[0];
		if(rctid==0)
			fb1=0;
		
		unsigned tmpid=0;
		for(unsigned i=0;i!=nreplica;++i)
		{
			if(i==rctid)
				fb[i]=-fb1;
			else
				fb[i]=new_coe[tmpid++]-fb1;
		}
	}
#endif
}

void ITS_Bias::output_fb()
{
	if(mcycle%fb_stride==0)
	{
#ifdef __PLUMED_HAS_DYNET
		if(use_talits)
			ofb.printField("ITERATE_METHOD","TALITS");
		else
#endif
			ofb.printField("ITERATE_METHOD","Traditional");
		ofb.printField("ITERATE_STEP",int(mcycle));
		ofb.printField("BOLTZMANN_CONSTANT",kB);
		ofb.printField("PESHIFT",peshift);
		ofb.printField("COEFFICIENT_TYPE",(is_set_temps?"TEMPERATURE":"RATIO"));
		ofb.printField("NREPLICA",int(nreplica));
		ofb.printField("TARGET_RATIO_ENERGY",ratio_energy);

		for(unsigned i=0;i!=nreplica;++i)
		{
			std::string id;
			Tools::convert(i,id);
			ofb.printField("index",id);
			if(is_set_ratios)
				ofb.printField("ratio",int_ratios[i]);
			else
				ofb.printField("temperature",int_temps[i]);
			ofb.printField("fb_value",fb[i]);
			ofb.printField("norm_value",norml[i]);
			ofb.printField("partition",partition[i]);
			ofb.printField();
		}
		ofb.printf("#!-----END-OF-FB-COEFFICIENTS-----\n\n");
		ofb.flush();
	}
	if(mcycle%fbtrj_stride==0)
	{
		if(fbtrj_output)
		{
			ofbtrj.fmtField(" %f");
			ofbtrj.printField("step",int(mcycle));
		}
		//~ if(norm_output)
		//~ {
			//~ onormtrj.fmtField(" %f");
			//~ onormtrj.printField("step",int(mcycle));
		//~ }
		if(fbtrj_output)
		{
			for(unsigned i=0;i!=nreplica;++i)
			{
				std::string id;
				Tools::convert(i,id);
				std::string fbid="FB"+id;
				
				if(fbtrj_output)
					ofbtrj.printField(fbid,fb[i]);

			}
			ofbtrj.printField();
			ofbtrj.flush();
		}
		//~ if(norm_output)
		//~ {
			//~ onormtrj.printField();
			//~ onormtrj.flush();
		//~ }
		//~ if(peshift_output)
		//~ {
			//~ opstrj.fmtField(" %f");
			//~ opstrj.printField("step",int(mcycle));
			//~ opstrj.printField("PESHIFT",peshift);
			//~ opstrj.printField();
			//~ opstrj.flush();
		//~ }
	}
	if(rct_output&&mcycle%rct_stride==0)
	{
		orct.fmtField(" %f");
		orct.printField("step",int(mcycle));
		for(unsigned i=0;i!=nreplica;++i)
		{
			std::string id;
			Tools::convert(i,id);
			std::string fbid="RCT"+id;
			if(rct_output)
				orct.printField(fbid,fb_rct[i]);
		}
		orct.printField();
		orct.flush();
	}
	
	if(bias_output&&mcycle%bias_stride==0)
		output_bias();
}

unsigned ITS_Bias::find_rw_id(double rwtemp,double& dtl,double& dth)
{
	if(rwtemp<templ)
		plumed_merror("the reweight temperature is lower than the minimal temperature of ITS");
	if(rwtemp>temph)
		plumed_merror("the reweight temperature is larger than the maximal temperature of ITS");
	
	unsigned rwid=0;
	for(unsigned i=0;i!=nreplica-1;++i)
	{
		dtl=rwtemp-int_temps[i];
		dth=int_temps[i+1]-rwtemp;
		if(dtl>=0&&dth>=0)
		{
			rwid=i;
			break;
		}
	}
	if(rwid+1==nreplica)
		plumed_merror("Can't find the fb value responds to the reweight temperature");
	
	return rwid;
}

double ITS_Bias::find_rw_fb(unsigned rwid,double dtl,double dth)
{
	double fbl=fb[rwid];
	double fbh=fb[rwid+1];
	double dfb=fbh-fbl;
	
	double rwfb;
	if(dtl<dth)
		rwfb=fbl+dfb*dtl/(dtl+dth);
	else
		rwfb=fbh-dfb*dth/(dtl+dth);
	return rwfb;
}

double ITS_Bias::find_rw_fb(double rwtemp)
{
	double dtl,dth;
	unsigned rwid=find_rw_id(rwtemp,dtl,dth);
	return find_rw_fb(rwid,dtl,dth);
}

void ITS_Bias::setupOFile(std::string& file_name, OFile& ofile, const bool multi_sim_single_files)
{
    ofile.link(*this);
    std::string fname=file_name;
    if(multi_sim_single_files)
    {
		unsigned int r=0;
		if(comm.Get_rank()==0)
			r=multi_sim_comm.Get_rank();
		comm.Bcast(r,0);
		if(r>0)
			fname="/dev/null";
		ofile.enforceSuffix("");
    }
    ofile.open(fname);
    ofile.setHeavyFlush();
}

void ITS_Bias::setupOFiles(std::vector<std::string>& fnames, std::vector<OFile*>& OFiles, const bool multi_sim_single_files)
{
	OFiles.resize(fnames.size(),NULL);
	for(unsigned i=0; i!=fnames.size();++i)
	{
		OFiles[i] = new OFile();
		OFiles[i]->link(*this);
		if(multi_sim_single_files)
		{
			unsigned int r=0;
			if(comm.Get_rank()==0)
				{r=multi_sim_comm.Get_rank();}
			comm.Bcast(r,0);
			if(r>0)
				{fnames[i]="/dev/null";}
			OFiles[i]->enforceSuffix("");
		}
		OFiles[i]->open(fnames[i]);
		OFiles[i]->setHeavyFlush();
	}
}

unsigned ITS_Bias::read_fb_file(const std::string& fname,double& _kB,double& _peshift)
{	
	IFile ifb;
    ifb.link(*this);
    if(use_mw)
		ifb.enforceSuffix("");
		
	if(!ifb.FileExist(fname))
		plumed_merror("Cannot find fb file " + fname );
		
    ifb.open(fname);
    
	ifb.allowIgnoredFields();
	
	std::string iter_method="Tradtional";
	
	unsigned _ntemp=nreplica;

	unsigned read_count=0;
	while(ifb)
	{
		if(ifb.FieldExist("ITERATE_METHOD"))
			ifb.scanField("ITERATE_METHOD",iter_method);

		if(ifb.FieldExist("ITERATE_STEP"))
		{
			int tmc=0;
			ifb.scanField("ITERATE_STEP",tmc);
			mcycle=tmc;
		}

		if(ifb.FieldExist("BOLTZMANN_CONSTANT"))
			ifb.scanField("BOLTZMANN_CONSTANT",_kB);

		if(ifb.FieldExist("PESHIFT"))
			ifb.scanField("PESHIFT",_peshift);
		
		std::string coe_type="TEMPERATURE";
		if(ifb.FieldExist("COEFFICIENT_TYPE"))
			ifb.scanField("COEFFICIENT_TYPE",coe_type);
		if(coe_type=="TEMPERATURE")
		{
			is_set_temps=true;
			is_set_ratios=false;
		}
		else if(coe_type=="RATIO")
		{
			is_set_temps=false;
			is_set_ratios=true;
		}
		else
			plumed_merror("unrecognized COEFFICIENT_TYPE "+coe_type);
			
		if(ifb.FieldExist("PESHIFT"))
			ifb.scanField("PESHIFT",_peshift);

		if(ifb.FieldExist("NREPLICA"))
		{
			int int_ntemp;
			ifb.scanField("NREPLICA",int_ntemp);
			_ntemp=int_ntemp;
		}
		
		if(ifb.FieldExist("TARGET_RATIO_ENERGY"))
			ifb.scanField("TARGET_RATIO_ENERGY",ratio_energy);
		
		int_ratios.resize(_ntemp);
		int_temps.resize(_ntemp);
		fb.resize(_ntemp);
		
		if(ifb)
		{
			if(!ifb.FieldExist("fb_value"))
				plumed_merror("cannot found \"fb_value\" in file \""+fb_input+"\"");
			if(is_set_temps&&!ifb.FieldExist("temperature"))
				plumed_merror("cannot found \"temperature\" in file \""+fb_input+"\"");
			if(is_set_ratios&&!ifb.FieldExist("ratio"))
				plumed_merror("cannot found \"ratio\" in file \""+fb_input+"\"");
		}
		else
			break;

		if(!is_const)
		{
			if(ifb.FieldExist("norm_value"))
				read_norm=true;
		}
		
		if(ifb.FieldExist("partition"))
		{
			partition.resize(_ntemp);
			partition_initial=true;
		}

		if(read_norm)
			norml.resize(_ntemp);
		
		double tmpfb;
		
		for(unsigned i=0;i!=_ntemp;++i)
		{
			ifb.scanField("fb_value",tmpfb);
			double tmpt;
			if(is_set_ratios)
			{
				ifb.scanField("ratio",tmpt);
				int_ratios[i]=tmpt;
				int_temps[i]=sim_temp*tmpt;
			}
			else
			{
				ifb.scanField("temperature",tmpt);
				int_temps[i]=tmpt;
				int_ratios[i]=tmpt/sim_temp;
			}
			
			if(read_norm)
			{
				double tmpnb;
				ifb.scanField("norm_value",tmpnb);
				norml[i]=tmpnb;
			}
			if(partition_initial)
			{
				double zk;
				ifb.scanField("partition",zk);
				partition[i]=zk;
			}
			fb[i]=tmpfb;
			
			ifb.scanField();
		}
		++read_count;
	}
	
	peshift=_peshift;
	
	if(is_set_ratios)
	{
		nreplica=_ntemp;
		ratiol=ratioh=int_ratios[0];
		for(unsigned i=1;i!=int_ratios.size();++i)
		{
			if(int_ratios[i]>ratioh)
				ratioh=int_ratios[i];
			if(int_ratios[i]<ratiol)
				ratiol=int_ratios[i];
		}
		temph=sim_temp*ratiol;
		templ=sim_temp*ratioh;
	}
	else
	{
		nreplica=_ntemp;
		templ=temph=int_temps[0];
		for(unsigned i=1;i!=int_temps.size();++i)
		{
			if(int_temps[i]>temph)
				temph=int_temps[i];
			if(int_temps[i]<templ)
				templ=int_temps[i];
		}
		ratioh=templ/sim_temp;
		ratiol=temph/sim_temp;
	}
	if(fabs(_kB/kB-1)>1.0e-8)
	{
		double rescale=kB/_kB;
		peshift*=rescale;
	}
	ifb.close();
	return read_count;
}

void ITS_Bias::change_peshift(double new_shift)
{
	fb_rescale(new_shift-peshift);
	peshift=new_shift;
	set_peshift_ratio();
}

void ITS_Bias::set_peshift_ratio()
{
	for(unsigned i=0;i!=nreplica-1;++i)
		peshift_ratio[i]=(betak[i]-betak[i+1])*peshift;
}

void ITS_Bias::output_bias()
{
	std::string smcycle;
	Tools::convert(int(mcycle),smcycle);
	std::string file=bias_file+smcycle+".data";
	
	obias.link(*this);
	obias.open(file);
	
	if(step==0)
	{
		obias.addConstantField("ITERATE_STEP");
		obias.addConstantField("BOLTZMANN_CONSTANT");
		obias.addConstantField("PESHIFT");
	}

	obias.printField("ITERATE_STEP",int(mcycle));
	obias.printField("BOLTZMANN_CONSTANT",kB);
	obias.printField("PESHIFT",peshift);

	double sfb=fb[0];
	for(unsigned i=1;i!=nreplica;++i)
		exp_added(sfb,fb[i]);

	for(unsigned i=0;i!=bias_bins;++i)
	{
		double energy=bias_min+d_pot*i;
		double senergy=energy+peshift;

		double _gfsum=-betak[0]*senergy+fb[0];
		double _bgfsum=_gfsum+std::log(betak[0]);
		for(unsigned j=1;j!=nreplica;++j)
		{
			double tmp=-betak[j]*senergy+fb[j];
			exp_added(_gfsum,tmp);
			exp_added(_bgfsum,tmp+std::log(betak[j]));
		}

		double eff_pot=-_gfsum/beta0+sfb;
		double bias_ener=eff_pot-senergy;
		double fscale=exp(_bgfsum-_gfsum)/beta0;
		obias.printField("original_potential",energy);
		obias.printField("bias_potential",bias_ener);
		obias.printField("effective_potential",eff_pot);
		obias.printField("force_scale",fscale);
		obias.printField();
	}
	obias.flush();
	obias.close();
}


#ifdef __PLUMED_HAS_DYNET

// training the parameter of WGAN
float ITS_Bias::update_wgan(const std::vector<float>& all_temps_sample,std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Dim xs_dim({1},batch_size);
	Dim xt_dim({1},ntarget);

	std::vector<float> batch_sample(batch_size);
	Expression x_sample=input(cg,xs_dim,&batch_sample);
	Expression x_target=input(cg,xt_dim,&temps_target);
	Expression p_target=input(cg,xt_dim,&temps_dis);

	Expression y_sample=nn_wgan.run(x_sample,cg);
	Expression y_target=nn_wgan.run(x_target,cg);
	Expression l_target=y_target*p_target;

	Expression loss_sample=mean_batches(y_sample);
	Expression loss_target=mean_batches(l_target);

	Expression loss_wgan=loss_sample-loss_target;
	
	double wloss=0;
	double loss;
	std::vector<std::vector<float>> vec_batch_sample;
	for(unsigned i=0;i!=nepoch;++i)
	{
		for(unsigned j=0;j!=batch_size;++j)
			batch_sample[j]=all_temps_sample[i*batch_size+j];
		vec_batch_sample.push_back(batch_sample);
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
		batch_sample=vec_batch_sample[i];
		vec_fw.push_back(as_vector(cg.forward(y_sample)));
	}
	
	return wloss;
}

// update the coeffients of the basis function
float ITS_Bias::update_bias(const std::vector<float>& all_energy_sample,const std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Expression efb = parameter(cg,parm_bias);
	
	Dim input_dim({1},batch_size);
	Dim fw_dim({1},batch_size);
	std::vector<float> batch_sample(batch_size);
	Expression E = input(cg,input_dim,&batch_sample);
	std::vector<Expression> vgE;
	
	for(unsigned i=0;i!=nreplica;++i)
	{
		if(i!=rctid)
			vgE.push_back(-1.0*betak[i]*E);
	}
	Expression egE = reshape(concatenate_cols(vgE), Dim({nreplica-1},batch_size));
	Expression egf = efb + egE;
	Expression lsegf = logsumexp_dim(egf,0);
	Expression egf0 = -1.0*betak[rctid]*E;
	Expression ueff = -1.0*logsumexp({egf0,lsegf});
	
	std::vector<float> fw_batch;
	Expression fw = input(cg,fw_dim,&fw_batch);

	Expression loss_ueff = fw * ( ueff - beta0 * E);
	Expression loss_mean = mean_batches(loss_ueff);

	double bloss=0;
	for(unsigned i=0;i!=nepoch;++i)
	{
		fw_batch=vec_fw[i];
		for(unsigned j=0;j!=batch_size;++j)
			batch_sample[j]=all_energy_sample[i*batch_size+j];
		bloss += as_scalar(cg.forward(loss_mean));
		cg.backward(loss_mean);
		train_bias->update();
	}
	bloss/=nepoch;
	
	return bloss;
}

#endif

}
}
