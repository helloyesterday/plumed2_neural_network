#ifdef __PLUMED_HAS_DYNET

/**
 * \file rnnlm-batch.h
 * \defgroup ffbuilders ffbuilders
 * \brief Feed forward nets builders
 *
 * An example implementation of a simple multilayer perceptron
 *
 */

#include "DynetTools.h"

namespace PLMD {
namespace dytools {

//~ using namespace dynet;

Activation activation_function(const std::string& a)
{
	if(a=="LINEAR"||a=="linear"||a=="Linear")
		return Activation::LINEAR;
	if(a=="RELU"||a=="relu"||a=="Relu"||a=="ReLU")
		return Activation::RELU;
	if(a=="ELU"||a=="elu"||a=="EXPLU"||a=="ExpLu")
		return Activation::ELU;
	if(a=="SMOOTH_ELU"||a=="Smooth_ELU"||a=="smooth_elu"||a=="SELU")
		return Activation::SMOOTH_ELU;
	if(a=="SIGMOID"||a=="sigmoid"||a=="Sigmoid")
		return Activation::SIGMOID;
	if(a=="SWISH"||a=="Swish"||a=="swish"||a=="silu"||a=="SILU"||a=="sil"||a=="SiL")
		return Activation::SWISH;
	if(a=="TANH"||a=="tanh"||a=="Tanh")
		return Activation::TANH;
	if(a=="ASINH"||a=="Asinh"||a=="asinh")
		return Activation::ASINH;
	if(a=="SOFTMAX"||a=="softmax"||a=="Softmax"||a=="SoftMax"||a=="SoftMAX")
		return Activation::SOFTMAX;
	if(a=="SOFTPLUS"||a=="softplus"||a=="Softplus"||a=="SoftPlus"||a=="SP")
		return Activation::SOFTPLUS;
	if(a=="SHIFTED_SOFTPLUS"||a=="shifted_softplus"||a=="Shifted_softplus"||
		a=="Shifted_Softplus"||a=="SSP")
		return Activation::SHIFTED_SOFTPLUS;
	if(a=="SCALED_SHIFTED_SOFTPLUS"||a=="scaled_shifted_softplus"||
		a=="Scaled_shifted_softplus"||a=="Scaled_Shifted_Softplus"||a=="SSSP")
		return Activation::SCALED_SHIFTED_SOFTPLUS;
	if(a=="SELF_NORMALIZING_SHIFTED_SOFTPLUS"||a=="self_normalizing_shifted_softplus"||
		a=="Self_normalizing_shifted_softplus"||a=="Self_Normalizing_Shifted_Softplus"||a=="SNSP")
		return Activation::SELF_NORMALIZING_SHIFTED_SOFTPLUS;
	if(a=="SELF_NORMALIZING_SMOOTH_ELU"||a=="Self_normalizing_ELU"||
		a=="self_normalizing_ELU"||a=="self_normalizing_elu"||a=="SNELU")
		return Activation::SELF_NORMALIZING_SMOOTH_ELU;
	if(a=="SELF_NORMALIZING_TANH"||a=="Self_normalizing_TANH"||
		a=="self_normalizing_tanh"||a=="SNtanh"||a=="SNtanh")
		return Activation::SELF_NORMALIZING_TANH;
	if(a=="SELF_NORMALIZING_ASINH"||a=="Self_normalizing_ASINH"||
		a=="self_normalizing_asinh"||a=="SNasinh"||a=="SNASINH")
		return Activation::SELF_NORMALIZING_ASINH;
	std::cerr<<"ERROR! Can't recognize the activation function "+a<<std::endl;
	exit(-1);
}

void dynet_initialization(unsigned random_seed)
{
	int cc=1;
	char pp[]="plumed";
	char *vv[]={pp};
	char** ivv=vv;
	dynet::DynetParams params = dynet::extract_dynet_params(cc,ivv,true);

	params.random_seed=random_seed;
	
	dynet::initialize(params);
}

  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model dynet::ParameterCollection to contain parameters
   * \param layers Layers description
   */
MLP::MLP(dynet::ParameterCollection& model,std::vector<Layer> layers)
{
    // Verify layers compatibility
    for (unsigned l = 0; l < layers.size() - 1; ++l) {
      if (layers[l].output_dim != layers[l + 1].input_dim)
        throw std::invalid_argument("Layer dimensions don't match");
    }

    // Register parameters in model
    for (Layer layer : layers) {
      append(model, layer);
    }
}

  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
void MLP::append(dynet::ParameterCollection& model, Layer layer)
{
    // Check compatibility
    if (LAYERS > 0)
    {
      if (layers[LAYERS - 1].output_dim != layer.input_dim)
        throw std::invalid_argument("Layer dimensions don't match");
      output_dim=layer.output_dim;
	}
	else
      input_dim=layer.input_dim;

    // Add to layers
    layers.push_back(layer);
    LAYERS++;
    // Register parameters
    dynet::Parameter W = model.add_parameters({layer.output_dim, layer.input_dim});
    dynet::Parameter b = model.add_parameters({layer.output_dim});
    params.push_back({W, b});
    unsigned nw = layer.output_dim * layer.input_dim;
    unsigned nb = layer.output_dim;
    params_size.push_back({nw,nb});
    params_num+=nw;
    params_num+=nb;
}

  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
dynet::Expression MLP::run(dynet::Expression& x,dynet::ComputationGraph& cg)
{
    // dynet::Expression for the current hidden state
    dynet::Expression h_cur = x;
    for (unsigned l = 0; l < LAYERS; ++l) {
      // Initialize parameters in computation graph
      dynet::Expression W = dynet::parameter(cg, params[l][0]);
      dynet::Expression b = dynet::parameter(cg, params[l][1]);
      // Aplly affine transform
      dynet::Expression a = dynet::affine_transform({b, W, h_cur});
      // Apply activation function
      dynet::Expression h = dy_act_fun(a, layers[l].activation);
      // Take care of dropout
      dynet::Expression h_dropped;
      if (layers[l].dropout_rate > 0) {
        if (dropout_active) {
          // During training, drop random units
          dynet::Expression mask = random_bernoulli(cg, {layers[l].output_dim}, 1 - layers[l].dropout_rate);
          h_dropped = cmult(h, mask);
        } else {
          // At test time, multiply by the retention rate to scale
          h_dropped = h * (1 - layers[l].dropout_rate);
        }
      } else {
        // If there's no dropout, don't do anything
        h_dropped = h;
      }
      // Set current hidden state
      h_cur = h_dropped;
    }

    return h_cur;
}
  
  /**
   * \brief Return the negative log likelihood for the (batched) pair (x,y)
   * \details For a batched input \f$\{x_i\}_{i=1,\dots,N}\f$, \f$\{y_i\}_{i=1,\dots,N}\f$, this computes \f$\sum_{i=1}^N \log(P(y_i\vert x_i))\f$ where \f$P(\textbf{y}\vert x_i)\f$ is modelled with $\mathrm{softmax}(MLP(x_i))$
   *
   * \param x Input batch
   * \param labels Output labels
   * \param cg Computation graph
   * \return dynet::Expression for the negative log likelihood on the batch
   */
dynet::Expression MLP::get_nll(dynet::Expression& x,std::vector<unsigned> labels,dynet::ComputationGraph& cg)
{
    // compute output
    dynet::Expression y = run(x, cg);
    // Do softmax
    dynet::Expression losses = pickneglogsoftmax(y, labels);
    // Sum across batches
    return sum_batches(losses);
}

  /**
   * \brief Predict the most probable label
   * \details Returns the argmax of the softmax of the networks output
   *
   * \param x Input
   * \param cg Computation graph
   *
   * \return Label index
   */
int MLP::predict(dynet::Expression& x,dynet::ComputationGraph& cg)
{
    // run MLP to get class distribution
    dynet::Expression y = run(x, cg);
    // Get values
    std::vector<float> probs = as_vector(cg.forward(y));
    // Get argmax
    unsigned argmax = 0;
    for (unsigned i = 1; i < probs.size(); ++i) {
      if (probs[i] > probs[argmax])
        argmax = i;
    }

    return argmax;
}

void MLP::clip(float left,float right,bool clip_last_layer)
{
	for(unsigned i=0;i!=params.size();++i)
	{
		if((i+1)!=params.size()||clip_last_layer)
		{
			for(unsigned j=0;j!=params[i].size();++j)
				params[i][j].get_storage().clip(left,right);
		}
	}
}

void MLP::clip_inplace(float left,float right,bool clip_last_layer)
{
	for(unsigned i=0;i!=params.size();++i)
	{
		if((i+1)!=params.size()||clip_last_layer)
		{
			for(unsigned j=0;j!=params[i].size();++j)
				params[i][j].clip_inplace(left,right);
		}
	}
}

std::vector<float> MLP::get_parameters()
{
	std::vector<float> param_values;
	for(unsigned i=0;i!=params.size();++i)
	{
		for(unsigned j=0;j!=params[i].size();++j)
		{
			std::vector<float> vv=as_vector(*params[i][j].values());
			param_values.insert(param_values.end(),vv.begin(),vv.end());
		}
	}
	return param_values;
}

void MLP::set_parameters(const std::vector<float>& param_values)
{
	if(param_values.size()<params_num)
	{
		std::cerr<<"ERROR! The number of the parameter overflow!"<<std::endl;
		exit(-1);
	}
	unsigned ival=0;
	for(unsigned i=0;i!=params.size();++i)
	{
		for(unsigned j=0;j!=params[i].size();++j)
		{
			std::vector<float> new_params;
			for(unsigned k=0;k!=params_size[i][j];++k)
				new_params.push_back(param_values[ival++]);
			params[i][j].set_value(new_params);
		}
	}
}


dynet::Expression MLP_CV::energy(dynet::ComputationGraph& cg,dynet::Expression& x)
{
	dynet::Expression inputs;
	if(_has_periodic)
	{
		dynet::Expression scale=dynet::input(cg,{ncv},&cvs_scale);
		dynet::Expression sx=dynet::cmult(x,scale);
		if(npids.size()==0)
			inputs=dynet::concatenate({dynet::cos(sx),dynet::sin(sx)});
		else
		{
			dynet::Expression px=dynet::select_rows(sx,&pids);
			dynet::Expression npx=dynet::select_rows(sx,&npids);
			inputs=dynet::concatenate({dynet::cos(px),dynet::sin(px),npx});
		}
	}
	else
		inputs=x;
	return energy_scale*nn.run(inputs,cg);
}

void MLP_CV::set_periodic(const std::vector<bool> _is_pcvs)
{
	if(_is_pcvs.size()!=ncv)
	{
		std::cerr<<"The size of std::vector _is_pcvs must be equal to the number of CVs"<<std::endl;
		std::exit(-1);
	}
	is_pcvs=_is_pcvs;
	pids.resize(0);
	npids.resize(0);
	ninput=ncv;
	num_pcv=0;
	for(unsigned i=0;i!=ncv;++i)
	{
		if(is_pcvs[i])
		{
			pids.push_back(i);
			_has_periodic=true;
			++ninput;
			++num_pcv;
		}
		else
			npids.push_back(i);
	}
}

void MLP_CV::set_hidden_layers(const std::vector<unsigned>& _hidden_layers,const std::vector<Activation>& _act_funs)
{
	if(_hidden_layers.size()!=_act_funs.size())
	{
		std::cerr<<"The size of std::vector _hidden_layers must be equal to the size of std::vector _act_funs"<<std::endl;
		std::exit(-1);
	}
	nhidden=_hidden_layers.size();
	hidden_layers=_hidden_layers;
	act_funs=_act_funs;
}

void MLP_CV::build_neural_network(dynet::ParameterCollection& pc)
{
	if(ncv==0)
	{
		std::cerr<<"In order to build neural network, the number of CVs must larger than 0"<<std::endl;
		std::exit(-1);
	}
	unsigned ldim=ninput;
	for(unsigned i=0;i!=nhidden;++i)
	{
		nn.append(pc,Layer(ldim,hidden_layers[i],act_funs[i],0));
		ldim=hidden_layers[i];
	}
	nn.append(pc,Layer(ldim,1,Activation::LINEAR,0));
}

WGAN::WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target):
nn(_nn),
bsize(_bsize),
ntarget(_ntarget),
clip_min(-0.01),
clip_max(0.01)
{
	if(nn.get_output_dim()!=1)
	{
		std::cerr<<"ERROR! the output dimension must be one!"<<std::endl;
		std::exit(-1);
	}
	set_expression(x_svalues,x_tvalues,p_target);
}

WGAN::WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues):
nn(_nn),
bsize(_bsize),
ntarget(_ntarget),
clip_min(-0.01),
clip_max(0.01)
{
	if(nn.get_output_dim()!=1)
	{
		std::cerr<<"ERROR! the output dimension must be one!"<<std::endl;
		std::exit(-1);
	}
	set_expression(x_svalues,x_tvalues);
}

void WGAN::set_expression(std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target)
{
	dynet::Dim xs_dim({nn.get_input_dim()},bsize);
	dynet::Dim xt_dim({nn.get_input_dim()},ntarget);
	dynet::Dim p_dim({1},ntarget);
	
	x_sample=input(cg,xs_dim,&x_svalues);
	x_target=input(cg,xt_dim,&x_tvalues);
	dynet::Expression target_weights=input(cg,p_dim,&p_target);
	
	y_sample=nn.run(x_sample,cg);
	y_target=nn.run(x_target,cg);
	
	dynet::Expression loss_sample=mean_elems(y_sample);
	// the target target distribution must be normalized!
	dynet::Expression loss_target=dot_product(y_target,target_weights);
	loss_expr=loss_sample-loss_target;
}

void WGAN::set_expression(std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues)
{
	dynet::Dim xs_dim({nn.get_input_dim()},bsize);
	dynet::Dim xt_dim({nn.get_input_dim()},ntarget);
	
	x_sample=input(cg,xs_dim,&x_svalues);
	x_target=input(cg,xt_dim,&x_tvalues);
	
	y_sample=nn.run(x_sample,cg);
	y_target=nn.run(x_target,cg);
	
	dynet::Expression loss_sample=mean_elems(y_sample);
	dynet::Expression loss_target=mean_elems(y_target);
	
	loss_expr=loss_sample-loss_target;
}

float WGAN::update(dynet::Trainer& trainer)
{
	float loss=as_scalar(cg.forward(loss_expr));
	cg.backward(loss_expr,true);
	trainer.update();
	return loss;
}

dynet::Trainer* new_traniner(const std::string& algorithm,dynet::ParameterCollection& pc,std::string& fullname)
{
	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		dynet::Trainer *trainer = new dynet::SimpleSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		dynet::Trainer *trainer = new dynet::CyclicalSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		dynet::Trainer *trainer = new dynet::MomentumSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		dynet::Trainer *trainer = new dynet::AdagradTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		dynet::Trainer *trainer = new dynet::AdadeltaTrainer(pc);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		dynet::Trainer *trainer = new dynet::RMSPropTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		dynet::Trainer *trainer = new dynet::AdamTrainer(pc);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		dynet::Trainer *trainer = new dynet::AmsgradTrainer(pc);
		return trainer;
	}
	return NULL;
}

dynet::Trainer* new_traniner(const std::string& algorithm,dynet::ParameterCollection& pc,const std::vector<float>& params,std::string& fullname)
{
	if(params.size()==0)
		return new_traniner(algorithm,pc,fullname);

	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		dynet::Trainer *trainer = new dynet::SimpleSGDTrainer(pc,params[0]);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		dynet::Trainer *trainer=NULL;
		if(params.size()<2)
		{
			std::cerr<<"ERROR! CyclicalSGD needs at least two learning rates"<<std::endl;
			exit(-1);
		}
		else if(params.size()==2)
			trainer = new dynet::CyclicalSGDTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new dynet::CyclicalSGDTrainer(pc,params[0],params[1],params[2]);
		else if(params.size()==4)
			trainer = new dynet::CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3]);
		else
			trainer = new dynet::CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3],params[4]);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::MomentumSGDTrainer(pc,params[0]);
		else
			trainer = new dynet::MomentumSGDTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::AdagradTrainer(pc,params[0]);
		else
			trainer = new dynet::AdagradTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::AdadeltaTrainer(pc,params[0]);
		else
			trainer = new dynet::AdadeltaTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::RMSPropTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new dynet::RMSPropTrainer(pc,params[0],params[1]);
		else
			trainer = new dynet::RMSPropTrainer(pc,params[0],params[1],params[2]);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::AdamTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new dynet::AdamTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new dynet::AdamTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new dynet::AdamTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		dynet::Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new dynet::AmsgradTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new dynet::AmsgradTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new dynet::AmsgradTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new dynet::AmsgradTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	return NULL;
}


}
}

#endif
