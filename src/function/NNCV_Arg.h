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

#ifndef __PLUMED_NNCV_Arg_h
#define __PLUMED_NNCV_Arg_h

#include "core/NNCV.h"
#include "Function.h"

namespace PLMD {
namespace function {

class NNCV_Arg :
	public Function,
	public NNCV
{
private:
	bool is_has_periodic;

	std::vector<bool> arg_is_periodic;
	
	std::vector<unsigned> periodic_id;
	std::vector<unsigned> non_periodic_id;
	
	std::vector<float> args;
	std::vector<float> arg_min;
	std::vector<float> arg_max;
	std::vector<float> arg_period;
	std::vector<float> arg_rescale;
	
	std::vector<std::string> arg_label;
public:
	static void registerKeywords(Keywords&);
	explicit NNCV_Arg(const ActionOptions&ao);
	
	void calculate();
	
	std::vector<float> get_input_values() {return args;}
	std::vector<float> get_input_layer();
	
	dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x);
	dynet::Expression input_reform(dynet::ComputationGraph& cg,const dynet::Expression& x);
};

}
}

#endif
#endif
