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

#include "Mol_GNN.h"
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"

#include <memory>
#include <cmath>
#include <random>

namespace PLMD {
namespace colvar {

using namespace dytools;

//+PLUMEDOC FUNCTION Mol_GNN
/*

*/
//+ENDPLUMEDOC

void Mol_GNN::registerKeywords(Keywords& keys) {
	Colvar::registerKeywords( keys );
	useCustomisableComponents(keys);
	keys.add("atoms","ATOMS","the atoms as the input of molecular graphic neural network");
	//~ keys.add("compulsory","CUTOFF","the cutoff distance");
	keys.add("compulsory","ACTIVE_FUNCTION","SSP","the activation function for the neural network");
	keys.add("compulsory","PARAMETERS_OUTPUT","GNN_parameters.data","the file to output the parameters of the neural network");
	keys.add("optional","PARAM_READ_FILE","the file to output the parameters of the neural network");
	keys.add("optional","ATOMS_TYPES","the types of input atoms. Default set is all the atoms has different types");
	keys.add("optional","TYPES_NUMBER","the total number of atom types.");
	keys.addFlag("NO_UPDATE",false,"do not update the parameters of neural networks. If you want to use the neural network without link with external bias, this flag MUST BE USE!");
}

Mol_GNN::Mol_GNN(const ActionOptions&ao):
	PLUMED_COLVAR_INIT(ao),
	_is_linked(false),
	firsttime(true)
{
	std::vector<AtomNumber> atoms;
	parseAtomList("ATOMS",atoms);
	plumed_massert(atoms.size()>1,"Number of specified atoms should be larger than 1!");
	natoms=atoms.size();
	
	parse("ACTIVE_FUNCTION",str_af);
	af=activation_function(str_af);

	parse("PARAMETERS_OUTPUT",param_output_file);
	parse("PARAM_READ_FILE",param_read_file);

	parseVector("ATOMS_TYPES",atoms_id);
	ntypes=natoms;
	
	parse("TYPES_NUMBER",ntypes);
	plumed_massert(ntypes>0,"TYPES_NUMBER must be larger than 0!");
	if(atoms_id.size()>0)
	{
		if(atoms_id.size()==1)
		{
			unsigned id=atoms_id[0];
			atoms_id.assign(natoms,id);
		}
		else if(atoms_id.size()==natoms)
		{
			for(unsigned i=0;i!=natoms;++i)
				plumed_massert(atoms_id[i]<ntypes,"the id of ATOMS_TYPES must be smaller than TYPES_NUMBER!");
		}
		else
			plumed_merror("the size of ATOMS_TYPES mismatch!");
	}
	else
	{
		atoms_id.resize(natoms);
		for(unsigned i=0;i!=natoms;++i)
			atoms_id[i]=i;
	}

	parseFlag("NO_UPDATE",_no_update);
	bool nopbc=!pbc;
	parseFlag("NOPBC",nopbc);
	pbc=!nopbc;

	checkRead();
	
	if(_no_update||(!_is_linked))
	{
		unsigned random_seed=0;
		//~ if(useMultipleWalkers())
		//~ {
			if(comm.Get_rank()==0)
			{
				if(multi_sim_comm.Get_rank()==0)
				{
					std::random_device rd;
					random_seed=rd();
				}
				multi_sim_comm.Barrier();
				multi_sim_comm.Bcast(random_seed,0);
			}
			comm.Barrier();
			comm.Bcast(random_seed,0);
		//~ }
		//~ else
		//~ {
			//~ if(comm.Get_rank()==0)
			//~ {
				//~ std::random_device rd;
				//~ random_seed=rd();
			//~ }
			//~ comm.Barrier();
			//~ comm.Bcast(random_seed,0);
		//~ }

		dynet_initialization(random_seed);
	}

	if(pbc)
		log.printf("  using periodic boundary conditions\n");
	else
		log.printf("  without periodic boundary conditions\n");
	log.printf("  with number of input atoms: %d\n",int(natoms));
	log.printf("  with number of atom types: %d\n",int(ntypes));
	log.printf("  with atoms id and types id: \n");
	for(unsigned i=0;i!=natoms;++i)
		log.printf("    ATOM %d with type id %d\n",int(atoms[i].serial()),int(atoms_id[i]));
	log.printf("  with activation function: %s\n",str_af.c_str());
	log.printf("  with parameter output file: %s\n",param_output_file.c_str());

	if(param_read_file.size()>0)
		log.printf("  with parameter read file: %s\n",param_read_file.c_str());

	addValueWithDerivatives();
	setNotPeriodic();
	requestAtoms(atoms);
}

void Mol_GNN::prepare()
{
	if(firsttime)
	{
		set_parameters();
		if(param_read_file.size()>0)
		{
			dynet::TextFileLoader loader(param_read_file);
			loader.populate(pc);
		}
		dynet::TextFileSaver saver(param_output_file);
		saver.save(pc);
		
		firsttime=false;
	}
}

void Mol_GNN::calculate()
{

	dynet::ComputationGraph cg;
	std::vector<std::vector<unsigned>> dis_id(natoms);
	std::vector<std::vector<Vector>> mat_vec(natoms);
	std::vector<dynet::Expression> dy_dis(natoms);
	std::vector<dynet::Expression> dy_rbf(natoms);
	std::vector<std::vector<float>> mat_dis(natoms);

	std::vector<float> coords;
	for(unsigned i=0;i!=natoms;++i)
	{
		Vector atom0=getPosition(i);
		if(!_no_update)
		{
			for(unsigned j=0;j!=3;++j)
				coords.push_back(atom0[j]);
		}
		
		for(unsigned j=i+1;j<natoms;++j)
		{
			Vector vec;
			if(pbc)
				vec=pbcDistance(atom0,getPosition(j));
			else
				vec=delta(atom0,getPosition(j));

			float dis=vec.modulo();
			
			long_term(cg,i,j);
			if(dis<cutoff)
			{
				mat_dis[i].push_back(dis);
				mat_dis[j].push_back(dis);
				dis_id[i].push_back(j);
				dis_id[j].push_back(i);
				mat_vec[i].push_back(vec);
				mat_vec[j].push_back(vec);
			}
		}
		dy_dis[i]=dynet::input(cg,{1,unsigned(dis_id[i].size())},&mat_dis[i]);
		dy_rbf[i]=calc_rbf(cg,dy_dis[i]);
	}

	dynet::Expression output=calc_energy(cg,dy_rbf,dis_id);
	cg.forward(output);
	cg.backward(output,true);
	std::vector<float> energy=dynet::as_vector(output.value());

	if((!_no_update)&&(_is_linked))
	{
		++steps;
		coordinates_record.push_back(coords);
		energy_record.push_back(energy[0]);
		if(steps%update_steps==0)
		{
			coordinates_record.resize(0);
			energy_record.resize(0);
		}
	}

	std::vector<Vector> deriv(getNumberOfAtoms());
	Tensor virial;
	for(unsigned i=0;i!=natoms;++i)
	{
		std::vector<float> grad(dynet::as_vector(dy_dis[i].gradient()));
		calc_deriv(i,grad,mat_dis[i],dis_id[i],mat_vec[i],deriv,virial);
	}
	
	for(unsigned i=0;i!=deriv.size();++i)
		setAtomsDerivatives(i,deriv[i]);

	setValue(energy[0]);
	setBoxDerivativesNoPbc();
}

dynet::Expression Mol_GNN::energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& atoms_coord,bool use_cutoff)
{
	std::vector<std::vector<dynet::Expression>> mat_dis(natoms);
	std::vector<dynet::Expression> vec_rbf(natoms);
	std::vector<std::vector<unsigned>> dis_id(natoms);
	for(unsigned i=0;i!=natoms;++i)
	{
		for(unsigned j=i+1;j<natoms;++j)
		{
			dynet::Expression dis=dynet::sqrt(squared_distance(
				atoms_coord[i],atoms_coord[j]));

			if(use_cutoff)
			{
				std::vector<float> vdis=dynet::as_vector(dis.value());
				if(vdis[0]<cutoff)
				{
					mat_dis[i].push_back(dis);
					mat_dis[j].push_back(dis);
					dis_id[i].push_back(j);
					dis_id[j].push_back(i);
				}
			}
			else
			{
				mat_dis[i].push_back(dis);
				mat_dis[j].push_back(dis);
				dis_id[i].push_back(j);
				dis_id[j].push_back(i);
			}
		}
		dynet::Expression cdis=dynet::concatenate_cols(mat_dis[i]);
		vec_rbf[i]=calc_rbf(cg,cdis);
	}
	
	return calc_energy(cg,vec_rbf,dis_id);
}

dynet::Expression Mol_GNN::energy(dynet::ComputationGraph& cg,const dynet::Expression& atoms_coord,bool use_cutoff)
{
	std::vector<std::vector<dynet::Expression>> mat_dis(natoms);
	std::vector<dynet::Expression> vec_rbf(natoms);
	std::vector<std::vector<unsigned>> dis_id(natoms);
	std::vector<dynet::Expression> acoord;
	
	for(unsigned i=0;i!=natoms;++i)
	{
		std::vector<unsigned> id={i};
		acoord.push_back(dynet::select_cols(atoms_coord,id));
	}

	for(unsigned i=0;i!=natoms;++i)
	{
		for(unsigned j=i+1;j<natoms;++j)
		{
			dynet::Expression dis=dynet::sqrt(squared_distance(
				acoord[i],acoord[j]));
				
			if(use_cutoff)
			{
				std::vector<float> vdis=dynet::as_vector(dis.value());
				if(vdis[0]<cutoff)
				{
					mat_dis[i].push_back(dis);
					mat_dis[j].push_back(dis);
					dis_id[i].push_back(j);
					dis_id[j].push_back(i);
				}
			}
			else
			{
				mat_dis[i].push_back(dis);
				mat_dis[j].push_back(dis);
				dis_id[i].push_back(j);
				dis_id[j].push_back(i);
			}
		}
		dynet::Expression cdis=dynet::concatenate_cols(mat_dis[i]);
		vec_rbf[i]=calc_rbf(cg,cdis);
	}
	
	return calc_energy(cg,vec_rbf,dis_id);
}

inline void Mol_GNN::calc_deriv(unsigned i,const std::vector<float>& grad,const std::vector<float>& dis,const std::vector<unsigned>& ids,const std::vector<Vector>& vec,std::vector<Vector>& deriv,Tensor& virial)
{
	unsigned id0=i;
	for(unsigned j=0;j!=ids.size();++j)
	{
		unsigned id1=ids[j];
		Vector dd((grad[j] / dis[j]) * vec[j]);
		Tensor vv(dd,vec[j]);
		deriv[id0]-=dd;
		deriv[id1]+=dd;
		virial-=vv;
	}
}


}
}

#endif
