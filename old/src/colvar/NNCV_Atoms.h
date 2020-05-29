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

#ifndef __DYNET_NNCV_ATOMS_H
#define __DYNET_NNCV_ATOMS_H

#include "core/NNCV.h"
#include "Colvar.h"

#define PLUMED_NNCV_ATOMS_INIT(ao) Action(ao),NNCV_Atoms(ao)

namespace PLMD {
namespace colvar {

//+PLUMEDOC FUNCTION NNCV_Atoms
/*

*/
//+ENDPLUMEDOC

class NNCV_Atoms :
	public Colvar,
	public NNCV
{
protected:
	bool is_pbc;
	std::vector<float> input_atoms;
public:
	explicit NNCV_Atoms(const ActionOptions&);
	virtual void calculate(){}
	static void registerKeywords(Keywords& keys);
	
	std::vector<float> get_input_data() {return input_atoms;}
};

}
}

#endif
#endif
