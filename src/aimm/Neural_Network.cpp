/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2018-2020 The AIMM code team
   (see the PEOPLE-AIMM file at the root of this folder for a list of names)

   See https://github.com/helloyesterday for more information.

   This file is part of AIMM code module.

   The AIMM code module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The AIMM code module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the AIMM code module.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#include "Neural_Network.h"
#include <dynet/io.h>

namespace PLMD {
namespace aimm {
	
void Neural_Network::registerKeywords(Keywords& keys) {
  Action::registerKeywords(keys);
}

Neural_Network::Neural_Network(const ActionOptions&ao):
  Action(ao),
  _is_cv_linked(false),
  cv_ptr(NULL)
{
}

void Neural_Network::load_parameter(const std::string& filename)
{
	dynet::TextFileLoader loader(filename);
	loader.populate(pc);
}

void Neural_Network::save_parameter(const std::string& filename)
{
	dynet::TextFileSaver savef(filename);
	savef.save(pc);
}


}
}

#endif
