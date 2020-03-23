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

#include "NNCV_Arg.h"
#include "ActionRegister.h"
#include "tools/Tools.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace function {

PLUMED_REGISTER_ACTION(NNCV_Arg,"NNCV_ARG")

void NNCV_Arg::registerKeywords(Keywords& keys) {
	Function::registerKeywords(keys);
	NNCV::registerKeywords(keys);
	
	keys.use("ARG");
	keys.add("optional","ARG_MIN","the minimal boundary of ARG");
	keys.add("optional","ARG_MAX","the maximal boundary of ARG");
	keys.add("optional","PERIODIC_ARG_ID","the ID of periodic ARG (the ID of first ARG is 0)");
	ActionWithValue::useCustomisableComponents(keys);
}

NNCV_Arg::NNCV_Arg(const ActionOptions&ao):
	Action(ao),
	Function(ao),
	NNCV(ao),
  	arg_is_periodic(getNumberOfArguments(),false),
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
	for(unsigned i=0;i!=getNumberOfArguments();++i)
	{
		arg_is_periodic[i]=getPntrToArgument(i)->isPeriodic();
		arg_label[i]=getPntrToArgument(i)->getName();
		if(arg_is_periodic[i])
			is_has_periodic=true;

		double minout,maxout;
		getPntrToArgument(i)->getDomain(minout,maxout);
		arg_min[i]=minout;
		arg_max[i]=maxout;
		arg_period[i]=maxout-minout;
	}
	ninput=getNumberOfArguments();
	
	std::vector<float> arg_min_in;
	std::vector<float> arg_max_in;
	parseVector("ARG_MIN",arg_min_in);
	parseVector("ARG_MAX",arg_max_in);
	
	std::vector<unsigned> periodic_arg_id;
	parseVector("PERIODIC_ARG_ID",periodic_arg_id);
	if(periodic_arg_id.size()>0)
	{
		is_has_periodic=true;
		arg_is_periodic.assign(getNumberOfArguments(),false);
		for(unsigned i=0;i!=periodic_arg_id.size();++i)
		{
			unsigned id=periodic_arg_id[i];
			plumed_massert(id<getNumberOfArguments(),"PERIODIC_ARG_ID must be smaller than the number of ARG");
			arg_is_periodic[id]=true;
		}
	}
	
	input_dim=ninput;
	for(unsigned i=0;i!=getNumberOfArguments();++i)
	{
		if(arg_is_periodic[i])
		{
			periodic_id.push_back(i);
			++input_dim;
		}
		else
			non_periodic_id.push_back(i);
	}
	set_input_dim();
	set_output_dim();
	build_neural_network();
	
	if(param_file.size()>0)
		load_parameter();
	
	if(arg_min_in.size()+arg_max_in.size()>0)
	{
		plumed_massert(arg_min_in.size()!=arg_max_in.size(),"the number of ARG_MAX must equal to ARG_MIN");
		if(arg_min_in.size()!=getNumberOfArguments())
		{
			if(arg_min_in.size()==1)
			{
				float min=arg_min_in[0];
				float max=arg_max_in[0];
				arg_min_in.assign(getNumberOfArguments(),min);
				arg_max_in.assign(getNumberOfArguments(),max);
			}
			else
				plumed_merror("the number of ARG_MAX and ARG_MIN mismatch with the number of ARG!");
		}
		for(unsigned i=0;i!=getNumberOfArguments();++i)
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
	
	log.printf("  using neural network as collective variable with %d arguments as input.\n",int(ninput));
	for(unsigned i=0;i!=getNumberOfArguments();++i)
	{
		if(arg_is_periodic[i])
			log.printf("    argument %d: \"%s\", periodic from %f to %f with period %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i],arg_period[i]);
		else
			log.printf("    argument %d: \"%s\", non-periodic from %f to %f.\n",int(i+1),arg_label[i].c_str(),arg_min[i],arg_max[i]);
	}
	log.printf("  with neural network model: %s.\n",nn_label.c_str());
	log.printf("  with input dimension for neural network: %d.\n",int(nn_ptr->get_input_dim()));
	log.printf("  with output dimension for neural network: %d.\n",int(nn_ptr->get_output_dim()));
}

void NNCV_Arg::calculate()
{
	for(unsigned i=0;i!=getNumberOfArguments();++i)
		args[i]=getArgument(i);
	
	dynet::ComputationGraph cg;
	dynet::Expression x=dynet::input(cg,{ninput},&args);
	dynet::Expression y=output(cg,x);
	std::vector<float> cv=dynet::as_vector(cg.forward(y));
	
	if(output_dim==1)
	{
		setValue(cv[0]);
		if(is_calc_deriv())
		{
			cg.backward(y,true);
			std::vector<float> deriv=dynet::as_vector(x.gradient());
			for(unsigned i=0;i!=getNumberOfArguments();++i)
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
				for(unsigned j=0;j!=getNumberOfArguments();++j)
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
		dynet::Expression scale=dynet::input(cg,{ninput},&arg_rescale);
		dynet::Expression sx=dynet::cmult(x,scale);
		if(non_periodic_id.size()==0)
			inputs=dynet::concatenate({dynet::cos(sx),dynet::sin(sx)});
		else
		{
			dynet::Expression px=dynet::select_rows(sx,&periodic_id);
			dynet::Expression npx=dynet::select_rows(sx,&non_periodic_id);
			inputs=dynet::concatenate({dynet::cos(px),dynet::sin(px),npx});
		}
		return inputs;
	}
	else
		return x;
}

std::vector<float> NNCV_Arg::get_input_layer()
{
	dynet::ComputationGraph cg;
	dynet::Expression x=dynet::input(cg,{input_dim},&args);
	dynet::Expression inputs=input_reform(cg,x);
	return dynet::as_vector(cg.forward(inputs));
}


}
}

#endif