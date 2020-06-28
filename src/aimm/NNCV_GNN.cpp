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
#include "NN_GNN.h"
#include "DynetTools.h"
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"

#include <memory>
#include <cmath>
#include <random>

namespace PLMD {
namespace aimm {

//+PLUMEDOC FUNCTION NNCV_GNN
/*

*/
//+ENDPLUMEDOC

class NNCV_GNN : public NNCV_Atoms
{
protected:
	unsigned natoms;
	unsigned ntypes;
	unsigned vec_dim;
	unsigned rbf_num;

	float cutoff;

	NN_GNN* gnn_ptr;

	std::vector<float> input_dis;
	std::vector<unsigned> atom_types;
		
	//~ void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j) {gnn_ptr->long_term(cg,i,j);}
	
	inline void calc_deriv(unsigned i,const std::vector<float>& grad,const std::vector<float>& dis,const std::vector<unsigned>& ids,const std::vector<Vector>& vec,std::vector<Vector>& deriv,Tensor& virial);

public:
	explicit NNCV_GNN(const ActionOptions&);
	void calculate();
	static void registerKeywords(Keywords& keys);
	
	bool calc_self_dis() const {return gnn_ptr->calc_self_dis();}
	
	void set_cutoff(float _cutoff){cutoff=_cutoff;}
	float get_cutoff() const {return cutoff;}
	
	void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j){gnn_ptr->long_term(cg,i,j);}
	
	std::vector<float> get_input_cvs() {return input_dis;}
	std::vector<float> get_input_layer() {return input_dis;}
	
	dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis) {return gnn_ptr->calc_rbf(cg,dis);}
	dynet::Expression gnn_output(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& neigh_rbf,const std::vector<std::vector<unsigned>>& neigh_id)
		{return gnn_ptr->gnn_output(cg,neigh_rbf,neigh_id);}
	
	dynet::Expression energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& atoms_coord,bool use_cutoff=true);
	dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x) {return nn_output(cg,x);}
};

PLUMED_REGISTER_ACTION(NNCV_GNN,"NNCV_GNN")

void NNCV_GNN::registerKeywords(Keywords& keys) {
	NNCV_Atoms::registerKeywords( keys );
	useCustomisableComponents(keys);
	keys.add("optional","CUTOFF","the cutoff distance");
	keys.add("optional","ATOMS_TYPES","the types of input atoms. Default set is all the atoms has different types");
}

NNCV_GNN::NNCV_GNN(const ActionOptions&ao):
	PLUMED_NNCV_ATOMS_INIT(ao),
	cutoff(0)
{
	gnn_ptr=plumed.getActionSet().selectWithLabel<NN_GNN*>(get_nn_label());
	if(!gnn_ptr)
		plumed_merror("The MODEL \""+get_nn_label()+"\" is not a graph neural network (NN_GNN).");
	
	parse("CUTOFF",cutoff);
	if(cutoff==0)
		cutoff=gnn_ptr->get_cutoff();
	
	natoms=getNumberOfAtoms();
	gnn_ptr->set_atoms_number(natoms);
	ntypes=gnn_ptr->get_atom_types_number();

	parseVector("ATOMS_TYPES",atom_types);
	if(atom_types.size()>0)
	{
		if(atom_types.size()==1)
		{
			unsigned id=atom_types[0];
			atom_types.assign(natoms,id);
		}
		else if(atom_types.size()==natoms)
		{
			if(ntypes>0)
			{
				for(unsigned i=0;i!=natoms;++i)
					plumed_massert(atom_types[i]<ntypes,"the id of ATOMS_TYPES must be smaller than TYPES_NUMBER!");
			}
		}
		else
			plumed_merror("the size of ATOMS_TYPES mismatch!");
	}
	else
	{
		atom_types.resize(natoms);
		for(unsigned i=0;i!=natoms;++i)
			atom_types[i]=i;
	}
	if(ntypes==0)
	{
		ntypes=atom_types.size();
		gnn_ptr->set_atom_types_number(ntypes);
	}
	
	gnn_ptr->set_atom_types(atom_types);
	
	ncvs=natoms*(natoms-1);
	input_dim=ncvs;
	
	set_input_dim();
	set_output_dim();

	checkRead();

	log.printf("  with cutoff distance: %f\n",cutoff);
	log.printf("  with number of input atoms: %d\n",int(natoms));
	log.printf("  with number of atom types: %d\n",int(ntypes));
	log.printf("  with atoms id and types id: \n");
	for(unsigned i=0;i!=natoms;++i)
		log.printf("    ATOM %d with type id %d\n",int(i),int(atom_types[i]));
}

void NNCV_GNN::calculate()
{
	std::vector<Vector> atoms_coord;
	input_atoms.resize(0);
	for(unsigned i=0;i!=natoms;++i)
	{
		Vector atom=getPosition(i);
		atoms_coord.push_back(atom);
		for(unsigned j=0;j!=3;++j)
			input_atoms.push_back(atom[j]);
	}
	
	std::vector<std::vector<unsigned>> neigh_id(natoms);
	std::vector<std::vector<Vector>> mat_vec(natoms);
	std::vector<std::vector<float>> dij(natoms);
	
	dynet::ComputationGraph cg;
	std::vector<dynet::Expression> dy_dis(natoms);
	std::vector<dynet::Expression> neigh_rbf(natoms);
	
	input_dis.resize(0);
	
	if(calc_self_dis())
	{
		Vector vec0(Vector(0,0,0));
		for(unsigned i=0;i!=natoms;++i)
		{
			Vector atom0=atoms_coord[i];
			for(unsigned j=i;j<natoms;++j)
			{
				if(j==i)
				{
					dij[i].push_back(0);
					neigh_id[i].push_back(j);
					mat_vec[i].push_back(vec0);
				}
				else
				{
					Vector atom1=atoms_coord[j];
					Vector vec;
					if(is_pbc)
						vec=pbcDistance(atom0,atom1);
					else
						vec=delta(atom0,atom1);

					float dis=vec.modulo();
					input_dis.push_back(dis);

					long_term(cg,i,j);
					if(dis<cutoff)
					{
						dij[i].push_back(dis);
						dij[j].push_back(dis);
						neigh_id[i].push_back(j);
						neigh_id[j].push_back(i);
						mat_vec[i].push_back(vec);
						mat_vec[j].push_back(vec);
					}
				}
			}
			dy_dis[i]=dynet::input(cg,{1,unsigned(neigh_id[i].size())},&dij[i]);
			neigh_rbf[i]=calc_rbf(cg,dy_dis[i]);
		}
	}
	else
	{
		for(unsigned i=0;i!=natoms;++i)
		{
			Vector atom0=atoms_coord[i];
			for(unsigned j=i+1;j<natoms;++j)
			{
				Vector atom1=atoms_coord[j];
				Vector vec;
				if(is_pbc)
					vec=pbcDistance(atom0,atom1);
				else
					vec=delta(atom0,atom1);

				float dis=vec.modulo();
				input_dis.push_back(dis);

				long_term(cg,i,j);
				if(dis<cutoff)
				{
					dij[i].push_back(dis);
					dij[j].push_back(dis);
					neigh_id[i].push_back(j);
					neigh_id[j].push_back(i);
					mat_vec[i].push_back(vec);
					mat_vec[j].push_back(vec);
				}
			}
			dy_dis[i]=dynet::input(cg,{1,unsigned(neigh_id[i].size())},&dij[i]);
			neigh_rbf[i]=calc_rbf(cg,dy_dis[i]);
		}
	}

	dynet::Expression pred=gnn_output(cg,neigh_rbf,neigh_id);
	std::vector<float> output=dynet::as_vector(cg.forward(pred));
	
	if(get_output_dim()==1)
	{
		setValue(output[0]);
		if(is_calc_deriv())
		{
			cg.backward(pred,true);
			std::vector<Vector> deriv(getNumberOfAtoms());
			Tensor virial;
			for(unsigned i=0;i!=natoms;++i)
			{
				std::vector<float> grad(dynet::as_vector(dy_dis[i].gradient()));
				calc_deriv(i,grad,dij[i],neigh_id[i],mat_vec[i],deriv,virial);
			}
			
			for(unsigned i=0;i!=deriv.size();++i)
				setAtomsDerivatives(i,deriv[i]);

			setBoxDerivatives(virial);
		}
	}
	else
	{
		for(int i=0;i!=getNumberOfComponents();++i)
		{
			Value* v=getPntrToComponent(i);
			v->set(output[i]);
			
			if(is_calc_deriv())
			{
				std::vector<float> comp_id(get_output_dim(),0);
				comp_id[i]=1;
				dynet::Expression vi=dynet::input(cg,{1,get_output_dim()},&comp_id);
				dynet::Expression yi=vi*pred;
				cg.forward(yi);
				cg.backward(yi,true);
				
				std::vector<Vector> deriv(getNumberOfAtoms());
				Tensor virial;
				for(unsigned j=0;j!=natoms;++j)
				{
					std::vector<float> grad(dynet::as_vector(dy_dis[j].gradient()));
					calc_deriv(j,grad,dij[j],neigh_id[j],mat_vec[j],deriv,virial);
				}
				
				for(unsigned j=0;j!=deriv.size();++j)
					setAtomsDerivatives(v,j,deriv[j]);
				
				setBoxDerivatives(v,virial);
			}
		}
	}
}

inline void NNCV_GNN::calc_deriv(unsigned i,const std::vector<float>& grad,const std::vector<float>& dis,const std::vector<unsigned>& ids,const std::vector<Vector>& vec,std::vector<Vector>& deriv,Tensor& virial)
{
	unsigned id0=i;
	for(unsigned j=0;j!=ids.size();++j)
	{
		unsigned id1=ids[j];
		if(id0!=id1)
		{
			Vector dd((grad[j] / dis[j]) * vec[j]);
			Tensor vv(dd,vec[j]);
			deriv[id0]-=dd;
			deriv[id1]+=dd;
			virial-=vv;
		}
	}
}


}
}

#endif
