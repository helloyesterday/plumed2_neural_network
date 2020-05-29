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

#include "NN_Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/Exception.h"
#include "tools/Matrix.h"
#include "tools/Random.h"
#include "tools/Tools.h"
#include <cstring>
#include "tools/File.h"
#include <iostream>
#include <iomanip>
#include <ctime>

namespace PLMD{
namespace aimm{

//+PLUMEDOC AIMM NN_BIAS
/*

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(NN_Bias,"NN_BIAS")

void NN_Bias::registerKeywords(Keywords& keys)
{
	Bias::registerKeywords(keys);
	keys.remove("ARG");
	keys.add("compulsory","ARG","the argument here must be ONE neural-network colvar variable (NNCV), and the output dimentsion must be ONE."); 
	keys.add("compulsory","SCALE","10","a factor to scale the output of neural network as bias potential. The base number is kBT."); 
	keys.add("optional","TEMP","the temerature to calculate kBT, default value is the simulation temperature.");
	keys.addFlag("MULTIPLE_WALKERS",false,"use multiple walkers");
}

NN_Bias::NN_Bias(const ActionOptions& ao):
	PLUMED_BIAS_INIT(ao)
{
	plumed_massert(getNumberOfArguments()==1,"NN_Bias can only accept ONE neural network collective variable (NNCV) as input argument");
	
	nncv_label=getPntrToArgument(0)->getName();
	nncv_ptr=plumed.getActionSet().selectWithLabel<NNCV*>(nncv_label);
	if(!nncv_ptr)
		plumed_merror("Neural network collective variable \""+nncv_label+"\" does not exist. ARG must be NNCV and NN_BIAS should always be defined AFTER NNCV.");
	nncv_ptr->linkBias(this);
	plumed_massert(nncv_ptr->get_output_dim()==1,"The output dimension of NNCV \""+nncv_label+"\" must be ONE when it used as the input of a bias potential.");
	plumed_massert(nncv_ptr->is_calc_deriv(),"The NNCV \""+nncv_label+"\" should not used the flag \"NO_DERIVATIVE\" when it used as the argument of a bias potential.");
	
	kB=plumed.getAtoms().getKBoltzmann();
	temp=-1;
	parse("TEMP",temp);
	if(temp>0)
		kBT=kB*temp;
	else
	{
		kBT=plumed.getAtoms().getKbT();
		temp=kBT/kB;
	}
	beta=1.0/kBT;
	
	parse("SCALE",scale_factor);
	energy_scale=scale_factor*kBT;
	
	parseFlag("MULTIPLE_WALKERS",use_mw);
	
	checkRead();
	
	if(use_mw)
		log.printf("  using multiple walkers: \n");
	log.printf("  with Boltzmann constant: %f\n",kB);
	log.printf("  with temeprature: %f\n",temp);
	log.printf("  with rescale base (kBT): %f\n",kBT);
	log.printf("  with rescale factor: %f\n",scale_factor);
	log.printf("  with the constant value to scale the output of neural network as bias potential: %f\n",energy_scale);
	log.printf("  with neural network collective variable as bias potential: %s\n",nncv_label.c_str());
}

NN_Bias::~NN_Bias()
{
}

void NN_Bias::calculate()
{
	double output=getArgument(0);
	double vbias=output*energy_scale;
	
	setBias(vbias);
	setOutputForce(0,-energy_scale);
}


}
}

#endif
