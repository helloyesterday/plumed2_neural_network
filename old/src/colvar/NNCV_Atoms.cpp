/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	 Copyright (c) 2011-2017 The plumed team
	 (see the PEOPLE file at the root of the distribution for a list of names)

	 See http://www.plumed.org for more information.

	 This file is part of plumed, version 2.

	 plumed is free software: you can redistribute it and/or modify
	 it under the terms of the GNU Lesser General Public License as published by
	 the Free Software Foundation, either version 3 of the License, or
	 (at your option) any later version.

	 plumed is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
	 GNU Lesser General Public License for more details.

	 You should have received a copy of the GNU Lesser General Public License
	 along with plumed.	If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#include "NNCV_Atoms.h"

namespace PLMD {
namespace colvar {

//+PLUMEDOC FUNCTION NNCV_Atoms
/*

*/
//+ENDPLUMEDOC

void NNCV_Atoms::registerKeywords(Keywords& keys) {
	Colvar::registerKeywords( keys );
	NNCV::registerKeywords( keys );
	useCustomisableComponents(keys);
	keys.add("atoms","ATOMS","the atoms as the input of molecular graphic neural network");
}

NNCV_Atoms::NNCV_Atoms(const ActionOptions&ao):
	Action(ao),
	Colvar(ao),
	NNCV(ao),
	is_pbc(true)
{
	use_atoms=true;
	use_args=false;
	
	bool nopbc=!is_pbc;
	parseFlag("NOPBC",nopbc);
	is_pbc=!nopbc;
	
	std::vector<AtomNumber> atoms;
	parseAtomList("ATOMS",atoms);
	plumed_massert(atoms.size()>1,"Number of specified atoms should be larger than 1!");
	
	ninput=getNumberOfAtoms()*3;
	input_atoms.resize(ninput);
	
	checkRead();
	
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

	requestAtoms(atoms);
	
	if(is_pbc)
		log.printf("  using periodic boundary conditions\n");
}


}
}

#endif
