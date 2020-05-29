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

#include "core/NNCV.h"
#include "Function.h"
#include "ActionRegister.h"
#include "tools/Tools.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace function {
	
//+PLUMEDOC FUNCTION NNCV_Arg
/*

*/
//+ENDPLUMEDOC
	
class NNCV_Arg :
	public Function,
	public NNCV
{
private:
	bool is_has_periodic;

	unsigned narg;

	std::vector<bool> arg_is_periodic;
	
	std::vector<unsigned> periodic_id;
	std::vector<unsigned> non_periodic_id;
	std::vector<unsigned> fourier_order;
	
	std::vector<float> args;
	std::vector<float> arg_min;
	std::vector<float> arg_max;
	std::vector<float> arg_period;
	std::vector<float> arg_rescale;
	std::vector<float> grid_bins;
	std::vector<float> grid_space;
	
	std::vector<std::vector<float>> grid_values;
	
	std::vector<std::string> arg_label;
public:
	static void registerKeywords(Keywords&);
	explicit NNCV_Arg(const ActionOptions&ao);
	
	void calculate();
	
	void write_grid_file(OFile& ogrid,const std::string& label);
	
	std::vector<float> get_input_data() {return args;}
	std::vector<float> get_input_cvs() {return args;}
	std::vector<float> get_input_layer();
	
	std::vector<bool> inputs_are_periodic() const {return arg_is_periodic;}
	std::vector<float> inputs_min() const {return arg_min;}
	std::vector<float> inputs_max() const {return arg_max;}
	std::vector<float> inputs_period() const {return arg_period;}
	
	bool input_is_periodic(unsigned id) const {return arg_is_periodic[id];}
	float input_min(unsigned id) const {return arg_min[id];}
	float input_max(unsigned id) const {return arg_max[id];}
	float input_period(unsigned id) const {return arg_period[id];}
	
	dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x);
	dynet::Expression input_reform(dynet::ComputationGraph& cg,const dynet::Expression& x);
};

PLUMED_REGISTER_ACTION(NNCV_Arg,"NNCV_ARG")

void NNCV_Arg::registerKeywords(Keywords& keys) {
	Function::registerKeywords(keys);
	NNCV::registerKeywords(keys);
	
	keys.use("ARG");
	keys.add("optional","ARG_MIN","the minimal boundary of ARG");
	keys.add("optional","ARG_MAX","the maximal boundary of ARG");
	keys.add("optional","GRID_BIN","the number of bins for the grid to output the neural network");
	keys.add("optional","FOURIER_ORDER","the orders of Fourier serires (\\sum_n=1^N {cos[n*2*pi*(s/P)]+sin[n*2*pi*(s/P)]}) to expand the CV for each argument(0 means expansion). For the perodic argument, the default value is 1.");
	ActionWithValue::useCustomisableComponents(keys);
}

NNCV_Arg::NNCV_Arg(const ActionOptions&ao):
	Action(ao),
	Function(ao),
	NNCV(ao),
	narg(getNumberOfArguments()),
  	arg_is_periodic(getNumberOfArguments(),false),
  	fourier_order(getNumberOfArguments(),0),
	args(getNumberOfArguments(),0),
	arg_min(getNumberOfArguments(),0),
	arg_max(getNumberOfArguments(),0),
	arg_period(getNumberOfArguments(),0),
	arg_rescale(getNumberOfArguments(),1),
	arg_label(getNumberOfArguments(),"")
{
	use_args=true;
	use_atoms=false;
	
	is_has_periodic=false;
	for(unsigned i=0;i!=narg;++i)
	{
		arg_is_periodic[i]=getPntrToArgument(i)->isPeriodic();
		arg_label[i]=getPntrToArgument(i)->getName();
		if(arg_is_periodic[i])
		{
			is_has_periodic=true;
			fourier_order[i]=1;
			double minout,maxout;
			getPntrToArgument(i)->getDomain(minout,maxout);
			arg_min[i]=minout;
			arg_max[i]=maxout;
			arg_period[i]=maxout-minout;
		}
	}
	ninput=narg;
	ncvs=ninput;
	
	std::vector<unsigned> fo;
	parseVector("FOURIER_ORDER",fo);
	if(fo.size()>0)
	{
		fourier_order=fo;
		if(fourier_order.size()!=narg)
		{
			if(fourier_order.size()==1)
			{
				unsigned order=fourier_order[0];
				fourier_order.assign(narg,order);
			}
			else
				plumed_merror("the number of FOURIER_ORDER mismatch with the number of arguments");
		}
		
		is_has_periodic=false;
		arg_is_periodic.assign(narg,false);
		for(unsigned i=0;i!=narg;++i)
		{
			if(fourier_order[i]>0)
			{
				is_has_periodic=true;
				arg_is_periodic[i]=true;
			}
		}
	}
	
	input_dim=0;
	for(unsigned i=0;i!=narg;++i)
	{
		if(fourier_order[i]==0)
		{
			non_periodic_id.push_back(i);
			++input_dim;
		}
		else
		{
			periodic_id.push_back(i);
			input_dim+=2*fourier_order[i];
		}
	}
	
	set_input_dim();
	set_output_dim();
	build_neural_network();
	
	std::vector<float> arg_min_in;
	if(keywords.exists("ARG_MIN"))
		parseVector("ARG_MIN",arg_min_in);
	std::vector<float> arg_max_in;
	if(keywords.exists("ARG_MIN"))
		parseVector("ARG_MAX",arg_max_in);
	
	if(param_in.size()>0)
		load_parameter();
	
	if(arg_min_in.size()+arg_max_in.size()>0)
	{
		plumed_massert(arg_min_in.size()!=arg_max_in.size(),"the number of ARG_MAX must equal to ARG_MIN");
		if(arg_min_in.size()!=narg)
		{
			if(arg_min_in.size()==1)
			{
				float min=arg_min_in[0];
				float max=arg_max_in[0];
				arg_min_in.assign(narg,min);
				arg_max_in.assign(narg,max);
			}
			else
				plumed_merror("the number of ARG_MAX and ARG_MIN mismatch with the number of ARG!");
		}
		for(unsigned i=0;i!=narg;++i)
		{
			arg_min[i]=arg_min_in[i];
			arg_max[i]=arg_max_in[i];
			plumed_massert(arg_max[i]>=arg_min[i],"ARG_MAX must be lager than ARG_MIN");
			arg_period[i]=arg_max[i]-arg_min[i];
			if(arg_is_periodic[i])
			{
				double diff=2*pi-arg_period[i];
				if(std::fabs(diff)>1e-6)
					arg_rescale[i]=2*pi/arg_period[i];
			}
		}
	}
	
	if(keywords.exists("GRID_BIN"))
		parseVector("GRID_BIN",grid_bins);
	else if(grid_wstride>0)
		grid_bins.push_back(100);
		
	if(grid_wstride>0)
	{		
		if(grid_bins.size()!=narg)
		{
			if(grid_bins.size()==1)
			{
				unsigned bin=grid_bins[0];
				grid_bins.assign(narg,bin);
			}
			else
				plumed_merror("the number of bins mismatch!");
		}
		
		for(unsigned i=0;i!=narg;++i)
		{
			float space=arg_period[i]/grid_bins[i];
			grid_space.push_back(space);
			if(arg_is_periodic[i])
				++grid_bins[i];
			std::vector<float> vec_val;
			for(unsigned j=0;j!=grid_bins[i];++j)
				vec_val.push_back(arg_min[i]+j*space);
			grid_values.push_back(vec_val);
		}
	}
	
	if(output_dim==1)
	{
		if(is_calc_deriv())
			addValueWithDerivatives();
		setNotPeriodic();
	}
	else
	{
		for(unsigned i=0;i!=output_dim;++i)
		{
			std::string s;
			Tools::convert(i,s);
			s="y["+s+"]";
			if(is_calc_deriv())
				addComponentWithDerivatives(s);
			getPntrToComponent(i)->setNotPeriodic();
		}
	}
	
	checkRead();
	
	log.printf("  using %d arguments as collective variables.\n",int(ncvs));
	for(unsigned i=0;i!=narg;++i)
	{
		if(arg_is_periodic[i])
			log.printf("    CV s%d: \"%s\", periodic from %f to %f with period %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i],arg_period[i]);
		else
			log.printf("    CV s%d: \"%s\", non-periodic from %f to %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i]);
	}
	log.printf("  using neural network with %d input values.\n",int(input_dim));
	int id=0;
	for(unsigned i=0;i!=narg;++i)
	{
		if(arg_is_periodic[i])
		{
			for(unsigned j=0;j!=fourier_order[i];++j)
			{
				log.printf("    Input value %d: cos( %f * s%d ).\n",++id,arg_rescale[i]*(j+1),int(i+1));
				log.printf("    Input value %d: sin( %f * s%d ).\n",++id,arg_rescale[i]*(j+1),int(i+1));
			}
		}
		else
			log.printf("    Input value %d: %f * s%d.\n",++id,arg_rescale[i],int(i+1));
	}
	
	log.printf("  with neural network model: %s.\n",nn_label.c_str());
	log.printf("  with input dimension for neural network: %d.\n",int(nn_ptr->get_input_dim()));
	log.printf("  with output dimension for neural network: %d.\n",int(nn_ptr->get_output_dim()));
}

void NNCV_Arg::calculate()
{
	for(unsigned i=0;i!=narg;++i)
		args[i]=getArgument(i);
	
	dynet::ComputationGraph cg;
	dynet::Expression x=dynet::input(cg,{ncvs},&args);
	dynet::Expression y=output(cg,x);
	std::vector<float> cv=dynet::as_vector(cg.forward(y));
	
	if(output_dim==1)
	{
		setValue(cv[0]);
		if(is_calc_deriv())
		{
			cg.backward(y,true);
			std::vector<float> deriv=dynet::as_vector(x.gradient());
			for(unsigned i=0;i!=narg;++i)
				setDerivative(i,deriv[i]);
		}
	}
	else
	{
		for(int i=0;i!=getNumberOfComponents();++i)
		{
			Value* v=getPntrToComponent(i);
			v->set(cv[i]);
			
			if(is_calc_deriv())
			{
				std::vector<float> comp_id(output_dim,0);
				comp_id[i]=1;
				dynet::Expression vi=dynet::input(cg,{1,output_dim},&comp_id);
				dynet::Expression yi=vi*y;
				cg.forward(yi);
				cg.backward(yi,true);
				std::vector<float> deriv=dynet::as_vector(x.gradient());
				for(unsigned j=0;j!=narg;++j)
					setDerivative(v,j,deriv[j]);
			}
		}
	}
}

dynet::Expression NNCV_Arg::output(dynet::ComputationGraph& cg,const dynet::Expression& x)
{
	dynet::Expression inputs=input_reform(cg,x);
	return nn_output(cg,inputs);
}

dynet::Expression NNCV_Arg::input_reform(dynet::ComputationGraph& cg,const dynet::Expression& x)
{
	if(is_has_periodic)
	{
		dynet::Expression inputs;
		dynet::Expression scale=dynet::input(cg,{ncvs},&arg_rescale);
		dynet::Expression sx=dynet::cmult(x,scale);
		
		std::vector<dynet::Expression> fourier_args;
		for(unsigned i=0;i!=narg;++i)
		{
			std::vector<unsigned> id={i};
			dynet::Expression si=dynet::select_rows(sx,id);
			
			if(fourier_order[i]==0)
				fourier_args.push_back(si);
			else
			{
				for(unsigned j=0;j!=fourier_order[i];++j)
				{
					fourier_args.push_back(dynet::cos((j+1)*si));
					fourier_args.push_back(dynet::sin((j+1)*si));
				}
			}
		}
		inputs=dynet::concatenate(fourier_args);
		
		return inputs;
	}
	else
		return x;
}

std::vector<float> NNCV_Arg::get_input_layer()
{
	dynet::ComputationGraph cg;
	dynet::Expression x=dynet::input(cg,{narg},&args);
	dynet::Expression inputs=input_reform(cg,x);
	return dynet::as_vector(cg.forward(inputs));
}

void NNCV_Arg::write_grid_file(OFile& ogrid,const std::string& label)
{
	dynet::ComputationGraph cg;
	
	std::vector<float> args(narg);
	dynet::Expression arg_inputs=dynet::input(cg,{narg},&args);
	dynet::Expression arg_output=output(cg,arg_inputs)*grid_output_scale+
		grid_output_shift;
	
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
			unsigned j=id[i];
			float val=grid_values[i][j];
			ogrid.printField(arg_label[i],val);
			args[i]=val;
		}
		std::vector<float> out=dynet::as_vector(cg.forward(arg_output));
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
