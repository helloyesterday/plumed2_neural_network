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
#ifndef __PLUMED_DyNet_Initialization_h
#define __PLUMED_DyNet_Initialization_h

#include "Action.h"
#include <string>

namespace PLMD {

class Action;

class DyNet_Init :
  public Action
{
protected:
	bool use_mw;
	int argc;
	unsigned seed;
	std::string str_argv;
public:
	static void registerKeywords(Keywords&);
	explicit DyNet_Init(const ActionOptions&ao);

	void apply() {};
	void calculate() {};
	
	bool is_multiple_walkers() const {return use_mw;}
	void set_seed(unsigned _seed) {seed=_seed;}
	unsigned get_seed() const {return seed;}
};

}

#endif
#endif
