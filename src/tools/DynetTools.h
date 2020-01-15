#ifdef __PLUMED_HAS_DYNET

#ifndef DYNET_TOOLS_H
#define DYNET_TOOLS_H

/**
 * \file rnnlm-batch.h
 * \defgroup ffbuilders ffbuilders
 * \brief Feed forward nets builders
 *
 * An example implementation of a simple multilayer perceptron
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/timing.h>
#include <dynet/expr.h>
#include <dynet/io.h>

namespace PLMD {
namespace dytools {

dynet::Trainer* new_traniner(const std::string& algorithm,dynet::ParameterCollection& pc,std::string& fullname);
dynet::Trainer* new_traniner(const std::string& algorithm,dynet::ParameterCollection& pc,const std::vector<float>& params,std::string& fullname);

/**
 * \ingroup ffbuilders
 * Common activation functions used in multilayer perceptrons
 */
enum Activation {
	LINEAR, /**< `LINEAR` : Identity function \f$x\longrightarrow x\f$ */
	RELU, /**< `RELU` : Rectified linear unit \f$x\longrightarrow \max(0,x)\f$ */
	ELU, /**< `ELU` : Exponential linear unit \f$x\longrightarrow \alpha*(e^{x}-1)\f$ */
	SMOOTH_ELU, /**< `SMOOTH_ELU` : Smooth ELU \f$x\longrightarrow log(e^{alpha}*e^{x}+1)-alpha\f$ */
	SIGMOID, /**< `SIGMOID` : Sigmoid function \f$x\longrightarrow \frac {1} {1+e^{-x}}\f$ */
	SWISH,  /**< `SWISH` : Swish function \f$x\longrightarrow \frac {x} {1+e^{-x}}\f$ */
	TANH, /**< `TANH` : Tanh function \f$x\longrightarrow \frac {1-e^{-2x}} {1+e^{-2x}}\f$ */
	ASINH, /**< `ASINH` : Inverse hyperbolic sine \f$x\longrightarrow asinh(x)\f$ */
	SOFTMAX, /**< `SOFTMAX` : Softmax function \f$\textbf{x}=(x_i)_{i=1,\dots,n}\longrightarrow \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j} })_{i=1,\dots,n}\f$ */
	SOFTPLUS, /**< `SOFTPLUS` : Softplus function \f$x\longrightarrow log(e^{x}+1)\f$ */
	SHIFTED_SOFTPLUS, /**< `SHIFTED_SOFTPLUS` : Shifted softplus function (SSP) \f$x\longrightarrow log(0.5*e^{x}+0.5)\f$ */
	SCALED_SHIFTED_SOFTPLUS, /**< `SCALED_SHIFTED_SOFTPLUS` : Scaled shifted softplus function (SSSP) \f$x\longrightarrow 2*log(0.5*e^{x}+0.5)\f$ */
	SELF_NORMALIZING_SHIFTED_SOFTPLUS, /**< `SELF_NORMALIZING_SHIFTED_SOFTPLUS` : Self normalizing softplus function (SNSP) \f$x\longrightarrow 1.875596256135042*log(0.5*e^{x}+0.5)\f$ */
	SELF_NORMALIZING_SMOOTH_ELU, /**< `SELF_NORMALIZING_SMOOTH_ELU` : Self normalizing smooth ELU (SNSELU) \f$x\longrightarrow 1.574030675714671*(log(e^{alpha}*e^{x}+1)-alpha)\f$ */
	SELF_NORMALIZING_TANH, /**< `SELF_NORMALIZING_TANH` : Self normalizing tanh (SNTANH) \f$x\longrightarrow 1.592537419722831*tanh(x)\f$ */
	SELF_NORMALIZING_ASINH /**< `SELF_NORMALIZING_ASINH` : Self normalizing asinh (SNASINH) \f$x\longrightarrow 1.256734802399369*asinh(x)\f$ */
};

Activation activation_function(const std::string& a);

inline dynet::Expression dy_log1p(dynet::Expression x)
{
	return dynet::log(x+1.0);
}

inline dynet::Expression dy_softplus(dynet::Expression x)
{
	return dy_log1p(dynet::exp(x));
}

inline dynet::Expression dy_shifted_softplus(dynet::Expression x)
{
	return dynet::log(0.5*dynet::exp(x)+0.5);
}

inline dynet::Expression dy_smooth_elu(dynet::Expression x)
{
	return dy_log1p(1.718281828459045*dynet::exp(x))-1.0;
}

inline dynet::Expression dy_act_fun(dynet::Expression h, Activation f)
{
	switch (f)
	{
	case LINEAR:
		return h;
		break;
	case RELU:
		return dynet::rectify(h);
		break;
	case ELU:
		return dynet::elu(h);
		break;
	case SMOOTH_ELU:
		return dy_smooth_elu(h);
		break;
	case SIGMOID:
		return dynet::logistic(h);
		break;
	case SWISH:
		return dynet::silu(h);
		break;
	case TANH:
		return dynet::tanh(h);
		break;
	case ASINH:
		return dynet::asinh(h);
		break;
	case SOFTMAX:
		return dynet::softmax(h);
		break;
	case SOFTPLUS:
		return dy_softplus(h);
		break;
	case SHIFTED_SOFTPLUS:
		return dy_shifted_softplus(h);
		break;
	case SCALED_SHIFTED_SOFTPLUS:
		return 2*dy_shifted_softplus(h);
		break;
	case SELF_NORMALIZING_SHIFTED_SOFTPLUS:
		return 1.875596256135042*dy_shifted_softplus(h);
		break;
	case SELF_NORMALIZING_SMOOTH_ELU:
		return 1.574030675714671*dy_smooth_elu(h);
		break;
	case SELF_NORMALIZING_TANH:
		return 1.592537419722831*dynet::tanh(h);
		break;
	case SELF_NORMALIZING_ASINH:
		return 1.256734802399369*dynet::asinh(h);
		break;
    default:
		throw std::invalid_argument("Unknown activation function");
		break;
	}
}

void dynet_initialization(unsigned random_seed);

/**
 * \ingroup ffbuilders
 * \struct Layer
 * \brief Simple layer structure
 * \details Contains all parameters defining a layer
 *
 */
struct Layer {
public:
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */
  Activation activation = LINEAR; /**< Activation function */
  float dropout_rate = 0; /**< Dropout rate */
  /**
   * \brief Build a feed forward layer
   *
   * \param input_dim Input dimension
   * \param output_dim Output dimension
   * \param activation Activation function
   * \param dropout_rate Dropout rate
   */
  Layer(unsigned input_dim, unsigned output_dim, Activation activation, float dropout_rate) :
    input_dim(input_dim),
    output_dim(output_dim),
    activation(activation),
    dropout_rate(dropout_rate) {};
  Layer() {};
};

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */
struct MLP {
protected:
  // Hyper-parameters
  unsigned LAYERS = 0;
  unsigned params_num = 0;
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */

  // Layers
  std::vector<Layer> layers;
  // Parameters
  std::vector<std::vector<dynet::Parameter>> params;
  std::vector<std::vector<unsigned>> params_size;

  bool dropout_active = true;

public:
  unsigned get_layers() const {return LAYERS;}
  unsigned get_output_dim() const {return input_dim;}
  unsigned get_input_dim() const {return output_dim;}
  void clip(float left,float right,bool clip_last_layer=false);
  void clip_inplace(float left,float right,bool clip_last_layer=false);
  
  unsigned parameters_number() const {return params_num;}
  
   /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  explicit MLP():LAYERS(0) {}
  
   /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  explicit MLP(dynet::ParameterCollection & model):LAYERS(0){}
  
  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model dynet::ParameterCollection to contain parameters
   * \param layers Layers description
   */
  explicit MLP(dynet::ParameterCollection& model,std::vector<Layer> layers);
  
  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
  void append(dynet::ParameterCollection& model, Layer layer);
  
    /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  dynet::Expression run(dynet::Expression x,dynet::ComputationGraph& cg);
                 
  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  dynet::Expression get_grad(dynet::Expression x,dynet::ComputationGraph& cg);
  
  /**
   * \brief Return the negative log likelihood for the (batched) pair (x,y)
   * \details For a batched input \f$\{x_i\}_{i=1,\dots,N}\f$, \f$\{y_i\}_{i=1,\dots,N}\f$, this computes \f$\sum_{i=1}^N \log(P(y_i\vert x_i))\f$ where \f$P(\textbf{y}\vert x_i)\f$ is modelled with $\mathrm{softmax}(MLP(x_i))$
   *
   * \param x Input batch
   * \param labels Output labels
   * \param cg Computation graph
   * \return dynet::Expression for the negative log likelihood on the batch
   */
  dynet::Expression get_nll(dynet::Expression x,std::vector<unsigned> labels,dynet::ComputationGraph& cg);
  
  /**
   * \brief Predict the most probable label
   * \details Returns the argmax of the softmax of the networks output
   *
   * \param x Input
   * \param cg Computation graph
   *
   * \return Label index
   */
  int predict(dynet::Expression x,dynet::ComputationGraph& cg);
  
    /**
   * \brief Enable dropout
   * \details This is supposed to be used during training or during testing if you want to sample outputs using montecarlo
   */
  void enable_dropout() {
    dropout_active = true;
  }

  /**
   * \brief Disable dropout
   * \details Do this during testing if you want a deterministic network
   */
  void disable_dropout() {
    dropout_active = false;
  }

  /**
   * \brief Check wether dropout is enabled or not
   *
   * \return Dropout state
   */
  bool is_dropout_enabled() {
    return dropout_active;
  }
  
  void set_parameters(const std::vector<float>&);
  std::vector<float> get_parameters();

private:
  //~ inline dynet::Expression activate(dynet::Expression h, Activation f);
  //~ inline dynet::Expression activate_grad(dynet::Expression h, Activation f);
};

class WGAN {
public:
	WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target);
	WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues);
	
	void clear_cg(){cg.clear();}

	unsigned batch_size() const {return bsize;}
	void set_batch_size(unsigned new_size) {bsize=new_size;}
	float update(dynet::Trainer& trainer);
private:
	dynet::ComputationGraph cg;
	MLP nn;
	unsigned bsize;
	unsigned ntarget;
	
	float clip_min;
	float clip_max;
	
	dynet::Expression x_sample;
	dynet::Expression x_target;
	dynet::Expression y_sample;
	dynet::Expression y_target;
	dynet::Expression loss_expr;
	
	void set_expression(std::vector<dynet::real>& x_svalues,
		std::vector<dynet::real>& x_tvalues,
		std::vector<dynet::real>& p_target);
	void set_expression(std::vector<dynet::real>& x_svalues,
		std::vector<dynet::real>& x_tvalues);
};


}
}

#endif

#endif
