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

#include "DyNet_Init.h"
#include "ActionRegister.h"
#include "tools/Communicator.h"
#include <random>
#include <dynet/init.h>

namespace PLMD {
	
PLUMED_REGISTER_ACTION(DyNet_Init,"DYNET_INIT")

void DyNet_Init::registerKeywords(Keywords& keys) {
  Action::registerKeywords(keys);
  keys.add("compulsory","SEED","-1","seed for the initialization of DyNet. -1 means use random seed");
  keys.addFlag("USE_MW",false,"use multiple walkers to perform MD simulation.");
}


DyNet_Init::DyNet_Init(const ActionOptions&ao):
  Action(ao),
  argc(1),
  str_argv("plumed")
{
	int _seed;
	parse("SEED",_seed);
	parseFlag("USE_MW",use_mw);
	
	unsigned sw_size;
	if(use_mw)
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				if(_seed<0)
				{
					std::random_device rd;
					seed=rd();
				}
				else 
					seed=_seed;
				sw_size=comm.Get_size();
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(seed,0);
			multi_sim_comm.Bcast(sw_size,0);
		}
		comm.Barrier();
		comm.Bcast(seed,0);
		comm.Bcast(sw_size,0);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			if(_seed<0)
			{
				std::random_device rd;
				seed=rd();
			}
			else 
				seed=_seed;
			sw_size=comm.Get_size();
		}
		comm.Barrier();
		comm.Bcast(seed,0);
		comm.Bcast(sw_size,0);
	}

	
	bool use_mpi=false;
	if(use_mw||sw_size>1)
		use_mpi=true;
	
	checkRead();
		
	std::string str_seed=std::to_string(seed);
	log.printf("  for DyNet C++ libaray initialization with seed: %s\n",str_seed.c_str());
	
	size_t len=str_argv.length();
	char *iargv=(char*)malloc((len+1)*sizeof(char));
	str_argv.copy(iargv,len,0);
	char *vargv[]={iargv};
	char** argv=vargv;
	
	dynet::DynetParams params = dynet::extract_dynet_params(argc,argv,use_mpi);
	params.random_seed=seed;
	dynet::initialize(params);
	
	delete iargv;
}


}

#endif
