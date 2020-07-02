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

#include "Mol_GNN.h"
#include "core/ActionRegister.h"

namespace PLMD {
namespace aimm {

//+PLUMEDOC FUNCTION SchNet
/*

*/
//+ENDPLUMEDOC

class SchNet :	public Mol_GNN
{
private:
	unsigned num_inlayers=3;
	unsigned num_dlayers=2;

	float rbf_width=0.3;
	float rbf_gamma;
	float scale_in_loss=0.01;
	
	std::vector<unsigned> out_aw_dims={32,1};
	
	void update_rbf_gamma(){rbf_gamma=1.0/(rbf_width*rbf_width);}
	void update_basis(){update_rbf_gamma();update_rbf_range();}
	void update_rbf_range(){
		float dr=cutoff/rbf_num;
		rbf_centers.resize(rbf_num);
		for(unsigned i=0;i!=rbf_num;++i)
			rbf_centers[i]=dr*(i+1);
	}
	
	void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j){}	
	
	dynet::Expression interaction(dynet::ComputationGraph& cg,const dynet::Expression& vxl,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id);
	
	dynet::Expression atom_wise(dynet::ComputationGraph& cg,const dynet::Expression& x){
		dynet::Expression W = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression b = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		return W*x+b;
	}
	
	dynet::Expression dense(dynet::ComputationGraph& cg,const dynet::Expression& r){
		dynet::Expression W = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression b = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		return actfun(W*r+b);
	}
	
	dynet::Expression filter_generator(dynet::ComputationGraph& cg,const dynet::Expression& rbf){
		dynet::Expression w=rbf;
		for(unsigned idense=0;idense!=num_dlayers;++idense)
			w=dense(cg,w);
		return w;
	}
	
	dynet::Expression read_out(dynet::ComputationGraph& cg,const dynet::Expression& x){
		dynet::Expression xl=x;
		for(unsigned iaw=0;iaw!=out_aw_dims.size();++iaw){
			dynet::Expression xa=atom_wise(cg,xl);	// atom-wise
			if(out_aw_dims[iaw]==1)	xl=xa;
			else xl=actfun(xa);	// shifted softplus
		}
		return xl;
	}

public:
	explicit SchNet(const ActionOptions&);
	static void registerKeywords(Keywords& keys);
	
	//~ dynet::Expression calc_energy(dynet::ComputationGraph& cg,const dynet::Expression& atoms_coord);
	dynet::Expression calc_energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id);
	
	void set_parameters();
	
	void set_rbf_width(float width){rbf_width=width;update_rbf_gamma();}
	
	float get_rbf_width() const {return rbf_width;}
	
	dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis){
		dynet::Expression gamma=dynet::const_parameter(cg, params[0][0]);
		dynet::Expression mu=dynet::const_parameter(cg, params[0][1]);
		dynet::Expression dis_vec=dynet::ones(cg,{rbf_num})*dis;
		dynet::Expression diff=dis_vec-mu;
		return dynet::exp(-dynet::cmult(gamma,dynet::square(diff)));
	}
};

PLUMED_REGISTER_ACTION(SchNet,"SCHNET")

void SchNet::registerKeywords(Keywords& keys) {
	Mol_GNN::registerKeywords( keys );
	keys.add("compulsory","CUTOFF","4.0","the cutoff distance");
	keys.add("compulsory","VECTOR_DIMENSION","64","the demension of the vector at the hidden layer of neural network of SchNet");
	keys.add("compulsory","RBF_NUMBER","300","the number of radical basis functions (RBF) to expand the distance bewteen each atoms");
	keys.add("compulsory","RBF_WIDTH","0.3","the width of gaussian used in RBF");
	keys.add("compulsory","INTER_NUMBER","3","the number of interaction layers of SchNet");
	keys.add("compulsory","DENSE_NUMBER","2","the number of dense blocks in the filter generator of SchNet");
}

SchNet::SchNet(const ActionOptions&ao):
  Action(ao),
  Mol_GNN(ao)
{
	double _cutoff;
	parse("CUTOFF",_cutoff);
	plumed_massert(_cutoff>0,"CUTOFF must be larger than 0!");
	cutoff=_cutoff;
	
	parse("VECTOR_DIMENSION",vec_dim);	
	parse("RBF_NUMBER",rbf_num);
	parse("INTER_NUMBER",num_inlayers);
	parse("DENSE_NUMBER",num_dlayers);
	
	double _rbf_width;
	parse("RBF_WIDTH",_rbf_width);
	plumed_massert(_rbf_width>0,"RBF_WIDTH must be larger than 0!");
	rbf_width=_rbf_width;
		
	log.printf("  with cutoff distance: %f\n",_cutoff);
	log.printf("  with number of radical basis functions: %d\n",int(rbf_num));
	log.printf("  with gaussian width at radical basis functions: %f\n",_rbf_width);
	log.printf("  with vector dimensions of neural network: %d\n",int(vec_dim));
	log.printf("  with number of interaction layers of neural network: %d\n",int(num_inlayers));
	log.printf("  with number of dense blocks of filter generator: %d\n",int(num_dlayers));
}

void SchNet::set_parameters()
{
	update_basis();
	params.resize(0);

	// calc_rbf
	dynet::Parameter gamma=pc.add_parameters({rbf_num});
	dynet::Parameter mu=pc.add_parameters({rbf_num});
	gamma.set_value(std::vector<float>(rbf_num,rbf_gamma));
	mu.set_value(rbf_centers);
	params.push_back({gamma,mu});
	
	// embedding
	dynet::Parameter a=pc.add_parameters({vec_dim,ntypes});
	params.push_back({a});
	
	// interaction layers
	for(unsigned l=0;l!=num_inlayers;++l)
	{
		// atom-wise 1
		dynet::Parameter aW=pc.add_parameters({vec_dim,vec_dim});
		dynet::Parameter ab=pc.add_parameters({vec_dim});
		params.push_back({aW,ab});
		
		// filter_generator: dense layers
		for(unsigned idense=0;idense!=num_dlayers;++idense)
		{
			unsigned input_dim=vec_dim;
			if(idense==0)
				input_dim=rbf_num;
			dynet::Parameter dW=pc.add_parameters({vec_dim,input_dim});
			dynet::Parameter db=pc.add_parameters({vec_dim});
			params.push_back({dW,db});
		}
		
		// atom-wise 2
		aW=pc.add_parameters({vec_dim,vec_dim});
		ab=pc.add_parameters({vec_dim});
		params.push_back({aW,ab});
		
		// atom-wise 3
		aW=pc.add_parameters({vec_dim,vec_dim});
		ab=pc.add_parameters({vec_dim});
		params.push_back({aW,ab});
	}
	
	// read out function
	unsigned input_dim=vec_dim;
	for(unsigned iaw=0;iaw!=out_aw_dims.size();++iaw)
	{
		// atom-wise
		unsigned output_dim=out_aw_dims[iaw];
		dynet::Parameter aW=pc.add_parameters({output_dim,input_dim});
		dynet::Parameter ab=pc.add_parameters({output_dim});
		input_dim=output_dim;
		params.push_back({aW,ab});
	}
}

dynet::Expression SchNet::calc_energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id)
{
	iparm=1;
	// embedding layer
	dynet::Expression x0=embedding(cg,atoms_id);
	
	//~ // interaction layer
	dynet::Expression vxl=x0;
	for(unsigned l=0;l!=num_inlayers;++l)
	{
		dynet::Expression vvl=interaction(cg,vxl,vec_rbf,dis_id);
		vxl=vxl+vvl;
	}
	
	// read out function
	dynet::Expression vy=read_out(cg,vxl);
	dynet::Expression pred=dynet::sum_dim(vy,{1});
	
	return pred;
}

dynet::Expression SchNet::interaction(dynet::ComputationGraph& cg,const dynet::Expression& vxl0,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id)
{
	// atom-wise
	unsigned iparm_begin=iparm;
	dynet::Expression vxl=atom_wise(cg,vxl0);
		
	std::vector<dynet::Expression> vec_cfconv;
	
	iparm_begin=iparm;
	for(unsigned iatom=0;iatom!=natoms;++iatom)
	{
		iparm=iparm_begin;
		
		dynet::Expression wl=filter_generator(cg,vec_rbf[iatom]);
		
		// continuous-filter convolutions
		dynet::Expression xl=dynet::select_cols(vxl,dis_id[iatom]);		
		dynet::Expression xl_wl=dynet::cmult(xl,wl);
		dynet::Expression cfconv=dynet::sum_dim(xl_wl,{1});
		vec_cfconv.push_back(cfconv);
	}
	dynet::Expression vcf=dynet::concatenate_cols(vec_cfconv);
		
	// atom-wise
	dynet::Expression vxa=atom_wise(cg,vcf);
		
	// shifted softplus
	dynet::Expression sxa=actfun(vxa);
		
	// atom-wise
	dynet::Expression vvl=atom_wise(cg,sxa);
	
	return vvl;
}

}
}

#endif
