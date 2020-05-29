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
#include "ActionRegister.h"

namespace PLMD {
namespace colvar {

using namespace dytools;

//+PLUMEDOC FUNCTION PhysNet
/*

*/
//+ENDPLUMEDOC

class PhysNet :	public Mol_GNN
{
private:
	unsigned nres_inter;
	unsigned nres_mod;
	unsigned nmod;
	unsigned output_dim;
	
	float rbf_scale;
	float rbf_beta;
	
	bool _is_calc_charege;
	
	std::vector<std::vector<dynet::Expression>> mat_dis_fun;
	void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j){
		if(_is_calc_charege)
			mat_dis_fun[i][j]=dynet::input(cg,{1},&mat_dis[i][j]);
	}
	
	void update_basis(){
		float expa=std::exp(-rbf_scale);
		float width=2.0*(1.0-expa)/rbf_num;
		rbf_beta=1.0/(width*width);
		float dmu=(1.0-expa)/(rbf_num-1);
		rbf_centers.resize(0);
		for(unsigned i=0;i!=rbf_num;++i)
			rbf_centers.push_back(1.0-i*dmu);
	}
	
	std::vector<dynet::Expression> module_layer(dynet::ComputationGraph& cg,
		const std::vector<dynet::Expression>& vxin,
		const std::vector<std::vector<dynet::Expression>>& rbf_mat,
		const std::vector<std::vector<unsigned>>& dis_id);
	std::vector<dynet::Expression> interaction(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vxl,const std::vector<std::vector<dynet::Expression>>& rbf_mat,const std::vector<std::vector<unsigned>>& dis_id);
	std::vector<dynet::Expression> run(dynet::ComputationGraph& cg,const std::vector<std::vector<dynet::Expression>>& rbf_mat,const std::vector<std::vector<unsigned>>& dis_id);
	
	// residual block
	dynet::Expression residual(dynet::ComputationGraph& cg,const dynet::Expression& x0,unsigned res_layers=2){
		dynet::Expression xl=x0;
		for(unsigned i=0;i!=res_layers;++i)
		{
			dynet::Expression W = dynet::parameter(cg, params[iparm][0]);
			dynet::Expression b = dynet::parameter(cg, params[iparm][1]);
			++iparm;
			xl = W * actfun(xl) + b;
		}
		return x0+xl;
	}
	
	void add_res_param(unsigned res_layers=2){
		for(unsigned i=0;i!=res_layers;++i){
			dynet::Parameter W=pc.add_parameters({vec_dim,vec_dim});
			dynet::Parameter b=pc.add_parameters({vec_dim});
			params.push_back({W,b});
		}
	}
	
	// output block
	dynet::Expression output_block(dynet::ComputationGraph& cg,const dynet::Expression& x){
		dynet::Expression xl=residual(cg,x);
		dynet::Expression W = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression b = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		return W * actfun(xl) + b;
	}
	
	// final prediction
	dynet::Expression final_pred(dynet::ComputationGraph& cg,const dynet::Expression& y){
		dynet::Expression s = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression z = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		return dynet::cmult(s,y)+z;
	}
	
	dynet::Expression cutoff_function(dynet::ComputationGraph& cg,const dynet::Expression& dis){
		std::vector<float> vdis=dynet::as_vector(dis.value());
		if(vdis[0]<cutoff)
		{
			dynet::Expression rc=dis/cutoff;
			dynet::Expression rc2=dynet::square(rc);
			dynet::Expression rc3=dynet::cube(rc);
			dynet::Expression rc4=dynet::square(rc2);
			dynet::Expression rc5=dynet::cmult(rc2,rc3);
			return -10.0*rc3+15.0*rc4-6.0*rc5+1.0;
		}
		else
			return dynet::zeros(cg,{1});
	}
	
	dynet::Expression dis_function(dynet::ComputationGraph& cg,dynet::Expression& dis){
		dynet::Expression dis2=dis*2;
		std::vector<float> vdis2=dynet::as_vector(dis2.value());
		if(vdis2[0]<cutoff)
			return dynet::ones(cg,{1})/dis;
		else
		{
			dynet::Expression phi2r=cutoff_function(cg,dis2);
			return phi2r*dynet::ones(cg,{1})/dynet::sqrt(dis*dis+1.0)+
				(dynet::ones(cg,{1})-phi2r)/dis;
		}
	}

public:
	explicit PhysNet(const ActionOptions&);
	static void registerKeywords(Keywords& keys);
	
	//~ dynet::Expression calc_energy(dynet::ComputationGraph& cg,const dynet::Expression& atoms_coord);
	dynet::Expression calc_energy(dynet::ComputationGraph& cg,const std::vector<std::vector<dynet::Expression>>& rbf_mat,const std::vector<std::vector<unsigned>>& dis_id);
	
	void set_parameters();
	
	void set_rbf_scale(float alpha){rbf_scale=alpha;update_basis();}
	void set_module_number(unsigned _nmod){nmod=_nmod;}
	
	float get_rbf_scale() const {return rbf_scale;}
	unsigned get_modoule_number() const {return nmod;}
	
	bool is_calc_charge() const {return _is_calc_charege;}
	
	dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis){
		dynet::Expression beta=dynet::const_parameter(cg, params[0][0]);
		dynet::Expression mu=dynet::const_parameter(cg, params[0][1]);
		dynet::Expression exp_dis=dynet::exp(-dis*rbf_scale/cutoff);
		dynet::Expression vec_exp_r=dynet::ones(cg,{rbf_num})*exp_dis;
		dynet::Expression diff=vec_exp_r-mu;
		dynet::Expression rbf=dynet::exp(-dynet::cmult(beta,dynet::square(diff)));
		return rbf*cutoff_function(cg,dis);
	}
};

PLUMED_REGISTER_ACTION(PhysNet,"PHYSNET")

void PhysNet::registerKeywords(Keywords& keys) {
	Mol_GNN::registerKeywords( keys );
	keys.add("compulsory","CUTOFF","1.0","the cutoff distance");
	keys.add("compulsory","VECTOR_DIMENSION","128","the demension of the vector at the hidden layer of neural network of SchNet");
	keys.add("compulsory","RBF_NUMBER","64","the number of radical basis functions (RBF) to expand the distance bewteen each atoms");
	keys.add("compulsory","RBF_SCALE","4","the const to scale the gaussian used in RBF");
	keys.add("compulsory","MODULE_NUMBER","5","the number of module layers of PhysNet");
	keys.add("compulsory","RESIDUAL_NUMBER_MODULE","2","the number of residual blocks in module layers");
	keys.add("compulsory","RESIDUAL_NUMBER_INTERACTION","3","the number of residual blocks in interaction layers");
	keys.addFlag("CALCULATE_CHARGE",false,"calcuate the charges of each atoms");
}

PhysNet::PhysNet(const ActionOptions&ao):
  Action(ao),
  Mol_GNN(ao),
  output_dim(1)
{
	double _cutoff;
	parse("CUTOFF",_cutoff);
	plumed_massert(_cutoff>0,"CUTOFF must be larger than 0!");
	cutoff=_cutoff;
	
	parse("VECTOR_DIMENSION",vec_dim);	
	parse("RBF_NUMBER",rbf_num);
	parse("MODULE_NUMBER",nmod);
	parse("RESIDUAL_NUMBER_MODULE",nres_mod);
	parse("RESIDUAL_NUMBER_INTERACTION",nres_inter);
	
	parseFlag("CALCULATE_CHARGE",_is_calc_charege);
	
	double _rbf_scale;
	parse("RBF_SCALE",_rbf_scale);
	rbf_scale=_rbf_scale;
	
	set_parameters();
	
	if(_is_calc_charege)
		log.printf("  with to calculate atom charges\n");
	else
		log.printf("  without to calculate atom charges\n");
	log.printf("  with cutoff distance: %f\n",_cutoff);
	log.printf("  with number of radical basis functions: %d\n",int(rbf_num));
	log.printf("  with constant to scale gaussian at radical basis functions: %f\n",_rbf_scale);
	log.printf("  with vector dimensions of neural network: %d\n",int(vec_dim));
	log.printf("  with number of module layers of neural network: %d\n",int(nmod));
	log.printf("  with number of residual blocks in module layers of neural network: %d\n",int(nres_mod));
	log.printf("  with number of residual blocks in interaction blocks of neural network: %d\n",int(nres_inter));
}

void PhysNet::set_parameters()
{
	update_basis();	
	params.resize(0);

	// calc_rbf
	dynet::Parameter beta=pc.add_parameters({rbf_num});
	dynet::Parameter mu=pc.add_parameters({rbf_num});
	beta.set_value(std::vector<float>(rbf_num,rbf_beta));
	mu.set_value(rbf_centers);
	params.push_back({beta,mu});
	
	// embedding layer
	dynet::Parameter a=pc.add_parameters({vec_dim,ntypes});
	params.push_back({a});
	
	output_dim=1;
	if(_is_calc_charege)
	{
		output_dim=2;
		mat_dis_fun.assign(natoms,std::vector<dynet::Expression>(natoms));
	}
	
	// module layer
	for(unsigned imod=0;imod!=nmod;++imod)
	{
		// interaction block
		dynet::Parameter G=pc.add_parameters({vec_dim,rbf_num});
		dynet::Parameter Wi=pc.add_parameters({vec_dim,vec_dim});
		dynet::Parameter bi=pc.add_parameters({vec_dim});
		dynet::Parameter Wj=pc.add_parameters({vec_dim,vec_dim});
		dynet::Parameter bj=pc.add_parameters({vec_dim});
		dynet::Parameter ul=pc.add_parameters({vec_dim});
		dynet::Parameter Wl=pc.add_parameters({vec_dim,vec_dim});
		dynet::Parameter bl=pc.add_parameters({vec_dim});
		params.push_back({G,Wi,bi,Wj,bj,ul,Wl,bl});
		
		// residual block at interaction block
		for(unsigned ires=0;ires!=nres_inter;++ires)
			add_res_param();
		
		// residual block at module layer
		for(unsigned ires=0;ires!=nres_mod;++ires)
			add_res_param();
		
		// output block
		add_res_param();
		dynet::Parameter oW=pc.add_parameters({output_dim,vec_dim});
		dynet::Parameter ob=pc.add_parameters({output_dim});
		params.push_back({oW,ob});
	}
	
	// final prediction
	dynet::Parameter s=pc.add_parameters({output_dim,natoms});
	dynet::Parameter z=pc.add_parameters({output_dim,natoms});
	params.push_back({s,z});
}

dynet::Expression PhysNet::calc_energy(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id)
{
	dynet::Expression vyfin(run(cg,vec_rbf,dis_id));
	
	dynet::Expression energy=dynet::zeros(cg,{1});
	if(_is_calc_charege)
	{
		std::vector<unsigned> ide={0};
		dynet::Expression ve=dynet::select_rows(vyfin,ide);
		
		std::vector<unsigned> idq={1};
		dynet::Expression vq=dynet::select_rows(vyfin,idq);
		
		energy=dynet::sum_elems(ve);
		
		dynet::Expression qenergy=dynet::zeros(cg,{1});
		for(unsigned i=0;i!=natoms;++i)
		{
			for(unsigned j=0;j!=i;++j)
			{
				dynet::Expression eq=vq[i]*vq[j]*mat_dis_fun[i][j];
				qenergy=qenergy+eq*2;
			}
		}
		energy=energy+qenergy;
	}
	else
	{
		energy=dynet::sum_dim(vyfin,{1});
	}
	return energy;
}

dynet::Expression PhysNet::run(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_rbf,const std::vector<std::vector<unsigned>>& dis_id)
{
	iparm=1;
	
	// embedding layer
	unsigned iparm_begin=iparm;
	dynet::Expression x0=embedding(cg,atoms_id);
	dynet::Expression vxl=x0;
	
	// module layer
	dynet::Expression vy=dynet::zeros(cg,{output_dim});
	for(unsigned imod=0;imod!=nmod;++imod)
	{
		dynet::Expression vxl = module_layer(cg,vxl,rbf_mat,dis_id);		
		dynet::Expression y = output_block(cg,vxl);
		vy = vy +y;
	}
	
	// final prediction
	dynet::Expression yfin=final_pred(cg,vy[iatom]);
	return vyfin;
}

dynet::Expression PhysNet::module_layer(dynet::ComputationGraph& cg,
	dynet::Expression& vxin,
	const std::vector<std::vector<dynet::Expression>>& rbf_mat,
	const std::vector<std::vector<unsigned>>& dis_id)
{
	// interaction layer
	dynet::Expression vxout=interaction(cg,vxin,rbf_mat,dis_id);
	
	// residual block
	dynet::Expression xl=vxout;
	for(unsigned ires=0;ires!=nres_mod;++ires)
		xl=residual(cg,xl);
	vxout=xl;
	return vxout;
}

// interaction block
dynet::Expression PhysNet::interaction(dynet::ComputationGraph& cg,
	const dynet::Expression& vxl,
	std::vector<dynet::Expression>& vec_rbf,
	const std::vector<std::vector<unsigned>>& dis_id)
{	
	dynet::Expression G = dynet::parameter(cg, params[iparm][0]);
	dynet::Expression Wi = dynet::parameter(cg, params[iparm][1]);
	dynet::Expression bi = dynet::parameter(cg, params[iparm][2]);
	dynet::Expression Wj = dynet::parameter(cg, params[iparm][3]);
	dynet::Expression bj = dynet::parameter(cg, params[iparm][4]);
	dynet::Expression ul = dynet::parameter(cg, params[iparm][5]);
	dynet::Expression Wl = dynet::parameter(cg, params[iparm][6]);
	dynet::Expression bl = dynet::parameter(cg, params[iparm][7]);
	++iparm;
	
	dynet::Expression vj_mat=actfun(Wj * actfun(vxl)+bj);
	
	std::vector<dynet::Expression> vvi1;
	for(unsigned iatom=0;iatom!=vec_rbf.size();++iatom)
	{
		std::vector<unsigned> id={i};
		dynet::Expression vi=dynet::select_cols(vj_mat,id);
		
		dynet::Expression gij = G * vec_rbf[iatom];
		dynet::Expression vs = dynet::select_cols(vj_mat,dis_id);
		dynet::Expression vij = dynet::cmult(gij,vs);
		
		dynet::Expression svij = dynet::sum_dim(vij,{1});
		dynet::Expression vi1 = vi + svij;
		vvi.push_back(vi1);
	}
	dynet::Expression vl=dynet::concatenate_cols(vvi);
	
	for(unsigned ires=0;ires!=nres_inter;++ires)
		vl=residual(cg,vl);
		
	dynet::Expression mul = ul * dynet::ones(cg,{1,natoms});
	dynet::Expression ul_xl = dynet::cmult(mul,vxl);
	dynet::Expression xl1 = ul_xl + Wl * actfun(vl) + bl;
	
	return vxl1;
}

}
}

#endif
