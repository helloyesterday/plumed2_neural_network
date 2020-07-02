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

#ifndef __PLUMED_NN_GNN_h
#define __PLUMED_NN_GNN_h

#include "Neural_Network.h"
#include "DynetTools.h"

namespace PLMD {
namespace aimm {

class NN_GNN :  public Neural_Network
{
protected:
	unsigned natoms;
	unsigned ntypes;
	unsigned iparm;
	unsigned vec_dim;
	unsigned rbf_num;
	
	bool pbc;
	bool firsttime;
	bool is_self_dis;
	
	float cutoff;
		
	std::string str_af;
	std::string param_read_file;
	std::string param_output_file;
	
	std::vector<unsigned> types_id;
	std::vector<unsigned> atom_types;
	std::vector<float> rbf_centers;
	
	std::vector<std::vector<dynet::Parameter>> params;
	
	Activation af;
	
	dynet::Expression actfun(const dynet::Expression& x){return dy_act_fun(x,af);}
	
	dynet::Expression embedding(dynet::ComputationGraph& cg){
		dynet::Expression a = dynet::parameter(cg, params[iparm++][0]);
		dynet::Expression zt = dynet::one_hot(cg,ntypes,atom_types);
		dynet::Expression vz = dynet::reshape(zt,{ntypes,natoms});
		return a*vz;
	}
	
	dynet::Expression embedding(dynet::ComputationGraph& cg,unsigned atom_id){
		dynet::Expression a = dynet::parameter(cg, params[iparm++][0]);
		dynet::Expression z = dynet::one_hot(cg,ntypes,atom_id);
		return a*z;
	}
	
	virtual void update_basis() {};
	
public:
	static void registerKeywords(Keywords&);
	explicit NN_GNN(const ActionOptions&ao);
	
	void calculate(){}
	void update(){}
	
	bool calc_self_dis() const {return is_self_dis;}
	
	void set_atom_types(const std::vector<unsigned> _atom_types) {atom_types=_atom_types;}
	void set_atoms_number(unsigned _natoms) {natoms=_natoms;}
	void set_atom_types_number(unsigned _ntypes) {ntypes=_ntypes;}
	void set_rbf_num(unsigned num){rbf_num=num;update_basis();}
	
	unsigned get_atom_types_number() const {return ntypes;}
	unsigned get_types_id(unsigned i) const {return types_id[i];}
	unsigned get_vector_dimension() const {return vec_dim;}
	unsigned get_rbf_num() const {return rbf_num;}
	unsigned get_cutoff() const {return cutoff;}
	void set_cutoff(float _cutoff){cutoff=_cutoff;update_basis();}
	
	virtual void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j){}
	
	unsigned parameters_number() const {return 0;}
	void set_parameters(const std::vector<float>& new_param) {}
	std::vector<float> get_parameters() const {return std::vector<float>();}
	
	void clip(float left,float right,bool clip_last=false) {}
	void clip_inplace(float left,float right,bool clip_last=false) {}
	
	virtual dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis) {return dynet::zeros(cg,{rbf_num,ntypes});}
	virtual dynet::Expression gnn_output(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& neigh_dis,const std::vector<std::vector<unsigned>>& neigh_id) {return dynet::zeros(cg,{output_dim});}
	
	virtual void build_neural_network() {}
	virtual dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x) {return dynet::zeros(cg,{output_dim});}
};

}
}

#endif
#endif
