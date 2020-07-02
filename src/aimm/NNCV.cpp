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

#include "NNCV.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"
#include "DynetTools.h"

namespace PLMD {
namespace aimm {
	
void NNCV::registerKeywords(Keywords& keys) {
	keys.add("compulsory","MODEL","the model to build neural network");
	keys.add("compulsory","OUTPUT_DIM","1","the dimension of the output of neural network"); 
    keys.add("optional","PARAM_READ_FILE","the file to read the parameter of neural network");
    keys.add("optional","PARAM_WSTRIDE","the frequency to output the parameter of neural network");
    keys.add("optional","PARAM_WRITE_FILE","the file to output the parameter of neural network");
    keys.add("optional","GRID_WRITE_FILE","the file to output the grid of the output of neural network");
    keys.add("optional","GRID_WSTRIDE","the frequency to output the grid of the output of neural network");
    keys.addFlag("NO_DERIVATIVE",false,"do not calculate the derivative of input w.r.t the neural network. DO NOT USE this flag when the CV is used for the input of an bias potential.");
}

NNCV::NNCV(const ActionOptions&ao):
  Action(ao),
  _is_nn_linked(false),
  _is_bias_linked(false),
  is_first(true),
  step(0),
  grid_output_scale(1),
  grid_output_shift(0),
  nn_ptr(NULL)
{
	_is_calc_deriv=true;
	bool is_no_deriv;
	parseFlag("NO_DERIVATIVE",is_no_deriv);
	_is_calc_deriv=!is_no_deriv;

	parse("MODEL",nn_label);
	nn_ptr=plumed.getActionSet().selectWithLabel<Neural_Network*>(nn_label);
	if(!nn_ptr)
		plumed_merror("Neural network \""+nn_label+"\" does not exist. NNCV should always be defined AFTER neural network.");
	nn_ptr->linkCV(this);
	_is_nn_linked=true;

	parse("OUTPUT_DIM",output_dim);
	plumed_massert(output_dim>0,"OUTPUT_DIM must be larger than 0!");	
	nn_ptr->set_output_dim(output_dim);
	
	if(keywords.exists("PARAM_READ_FILE"))
		parse("PARAM_READ_FILE",param_in);
		
	if(keywords.exists("PARAM_WRITE_FILE"))
		parse("PARAM_WRITE_FILE",param_out);
	param_wstride=0;
	if(keywords.exists("PARAM_WSTRIDE"))
		parse("PARAM_WSTRIDE",param_wstride);
	if(param_wstride>0&&param_out.size()==0)
		plumed_merror("GRID_WRITE_FILE is not define and the parameter file cannot be output!");

	if(keywords.exists("GRID_WRITE_FILE"))
		parse("GRID_WRITE_FILE",grid_file);
	grid_wstride=0;
	if(keywords.exists("GRID_WSTRIDE"))
		parse("GRID_WSTRIDE",grid_wstride);
	if(grid_wstride>0&&grid_file.size()==0)
		plumed_merror("GRID_WRITE_FILE is not define and the grid file cannot be output!");
	
	if(is_no_deriv)
		log.printf("  without calculating derivative\n");
	log.printf("  with output dimension: %d\n",int(output_dim));
	if(param_in.size())
		log.printf("  reading parameter of neural network from file: %s\n",param_in.c_str());
	if(param_wstride>0)
	{
		log.printf("  write the parameter of neural network to file: %s\n",param_out.c_str());
		log.printf("  with the output frequency: %d\n",param_wstride);
	}
}

NNCV::~NNCV()
{
	if(grid_wstride>0)
		ogrid.close();
}

void NNCV::update()
{
	if(param_wstride>0&&step%param_wstride==0)
		save_parameter();
		
	if(grid_wstride>0&&step%grid_wstride==0)
		write_grid_file();
	
	++step;
}

void NNCV::prepare()
{
	if(is_first)
	{
		build_neural_network();
		
		if(param_in.size()>0)
			load_parameter();

		is_first=false;
		if(!_is_bias_linked&&grid_wstride>0)
			ogrid_init(grid_file,false);
	}
}

void NNCV::ogrid_init(const std::string& filename,bool enforce_suffix)
{
	ogrid.link(*this);
	if(enforce_suffix)
		ogrid.enforceSuffix("");
	ogrid.open(filename.c_str());
	ogrid.fmtField(" %f");
}

}
}

#endif
