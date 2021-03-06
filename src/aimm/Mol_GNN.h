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

#ifndef __DYNET_MOL_GNN_H
#define __DYNET_MOL_GNN_H

#include "colvar/Colvar.h"
#include "DynetTools.h"

namespace PLMD {
namespace aimm {

//+PLUMEDOC FUNCTION Mol_GNN
/*

*/
//+ENDPLUMEDOC

class Mol_GNN :	public colvar::Colvar
{
protected:
	unsigned natoms;
	unsigned ntypes;
	unsigned iparm;
	unsigned vec_dim;
	unsigned rbf_num;
	unsigned steps;
	unsigned update_steps;
	
	bool _is_linked;
	bool _no_update;
	bool pbc;
	bool firsttime;

	float cutoff;
	
	std::string str_af;
	std::string param_read_file;
	std::string param_output_file;
	
	Activation af;

	std::vector<float> atom_coordinates;
	std::vector<std::vector<float>> coordinates_record;
	std::vector<float> energy_record;
	std::vector<float> rbf_centers;
	std::vector<unsigned> atoms_id;
	std::vector<std::vector<float>> atom_coord;
	//~ std::vector<std::vector<std::vector<float>>> mat_dis;
	std::vector<std::vector<dynet::Parameter>> params;
	
	dynet::ParameterCollection pc;
	
	virtual void update_basis()=0;
	virtual void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j)=0;

	dynet::Expression actfun(const dynet::Expression& x){return dy_act_fun(x,af);}

	dynet::Expression embedding(dynet::ComputationGraph& cg,unsigned atom_id){
		dynet::Expression a = dynet::parameter(cg, params[iparm++][0]);
		dynet::Expression z = dynet::one_hot(cg,ntypes,atom_id);
		return a*z;
	}
	
	dynet::Expression embedding(dynet::ComputationGraph& cg,const std::vector<unsigned>& atoms_id){
		dynet::Expression a = dynet::parameter(cg, params[iparm++][0]);
		dynet::Expression z = dynet::one_hot(cg,ntypes,atoms_id);
		dynet::Expression vz = dynet::reshape(z,{ntypes,natoms});
		return a*vz;
	}

public:
	explicit Mol_GNN(const ActionOptions&);
	virtual void calculate();
	virtual void prepare();
	static void registerKeywords(Keywords& keys);
	
	dynet::ParameterCollection& param_collect() {return pc;}
	virtual void set_parameters()=0;
	
	bool is_linked() const {return _is_linked;}
	bool no_update() const {return _no_update;}
	
	void set_rbf_num(unsigned num){rbf_num=num;update_basis();}
	void set_cutoff(float _cutoff){cutoff=_cutoff;update_basis();}
	
	unsigned get_rbf_num() const {return rbf_num;}
	float get_cutoff() const {return cutoff;}
	
	virtual dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis)=0;
	virtual dynet::Expression calc_energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id)=0;
	
	inline void calc_deriv(unsigned i,const std::vector<float>& grad,const std::vector<float>& dis,const std::vector<unsigned>& ids,const std::vector<Vector>& vec,std::vector<Vector>& deriv,Tensor& virial);
	
	dynet::Expression energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& atoms_coord,bool use_cutoff=true);
	dynet::Expression energy(dynet::ComputationGraph& cg,const dynet::Expression& atoms_coord,bool use_cutoff=false);
};

}
}

#endif
#endif
