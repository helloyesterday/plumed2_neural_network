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

#include "NN_GNN.h"
#include "DynetTools.h"
#include "core/ActionRegister.h"

namespace PLMD {
namespace aimm {
	
//+PLUMEDOC AIMM GNN_SCHNET
/*

*/
//+ENDPLUMEDOC
	
class GNN_AirNet :  public NN_GNN
{
private:
	unsigned num_inlayers;
	unsigned num_dlayers;
	//~ unsigned num_mlayers;
	unsigned att_layers;
	unsigned att_dim;
	unsigned max_cycles;
	unsigned ponder_layers;

	float rbf_sigma;
	float rbf_scale;
	float rbf_min;
	float rbf_log_min;
	float scale_in_loss=0.01;
	float weight_scale;
	
	std::vector<unsigned> ponder_dim;
	std::vector<unsigned> out_aw_dims={32,1};
	std::vector<std::vector<unsigned>> row_ids;
	
	std::vector<Activation> ponder_act_funs;
	
	void update_rbf_scale(){rbf_scale=0.5/(rbf_sigma*rbf_sigma);}
	void update_basis(){update_rbf_scale();update_rbf_range();}
	void update_rbf_range(){
		rbf_log_min=std::log(rbf_min/cutoff);
		float dmu=-rbf_log_min/(rbf_num-1);
		rbf_centers.resize(rbf_num);
		for(unsigned i=0;i!=rbf_num;++i)
			rbf_centers[i]=rbf_log_min+dmu*i;
	}
	
	dynet::Expression cutoff_function(dynet::ComputationGraph& cg,const dynet::Expression& dis){
		dynet::Expression rc=dis/cutoff;
		dynet::Expression rc2=dynet::square(rc);
		dynet::Expression rc3=dynet::cube(rc);
		dynet::Expression rc4=dynet::square(rc2);
		dynet::Expression rc5=dynet::cmult(rc2,rc3);
		dynet::Expression calc=-10.0*rc3+15.0*rc4-6.0*rc5+1.0;
		
		dynet::Expression zero=dynet::zeros(cg,{dis.dim()});
		return dynet::max(calc,zero);
	}
	
	void long_term(dynet::ComputationGraph& cg,unsigned i,unsigned j){}	
	
	void interaction(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vgl,const std::vector<std::vector<unsigned>>& neigh_id,std::vector<dynet::Expression>& vec_xi);
	void positional_embedding(dynet::ComputationGraph& cg,const dynet::Expression& xi,const dynet::Expression& xij,const dynet::Expression& gij,dynet::Expression& qi,dynet::Expression& Ki,dynet::Expression& Vi);
	dynet::Expression multi_head_attention(dynet::ComputationGraph& cg,const dynet::Expression& qi,const dynet::Expression& Ki,const dynet::Expression& Vi);
	dynet::Expression pondering(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vxl);
	
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
	
	dynet::Expression layer_normalization(dynet::ComputationGraph& cg,const dynet::Expression& x){
		dynet::Expression G = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression u = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		return dynet::layer_norm(x,G,u);
	}
	
	//~ dynet::Expression feedback_forward(dynet::ComputationGraph& cg,const dynet::Expression& x){
		//~ dynet::Expression y=x;
		//~ for(unsigned iaw=0;iaw!=num_mlayers;++iaw)
			//~ y=atom_wise(cg,y);
		//~ return y;
	//~ }
	
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
	static void registerKeywords(Keywords&);
	explicit GNN_AirNet(const ActionOptions&ao);
	
	void set_rbf_sigma(float sigma){rbf_sigma=sigma;update_rbf_scale();}
	
	float get_rbf_sigma() const {return rbf_sigma;}
	
	dynet::Expression calc_rbf(dynet::ComputationGraph& cg,const dynet::Expression& dis) {
		dynet::Expression scale=dynet::const_parameter(cg, params[0][0]);
		dynet::Expression mu=dynet::const_parameter(cg, params[0][1]);
		
		dynet::Expression dmin=dynet::constant(cg,{dis.dim()},rbf_min);
		dynet::Expression dis0=dynet::max(dis,dmin);
		
		dynet::Expression dcut=dynet::log(dis0/cutoff);
		dynet::Expression diff=dcut-mu;
		dynet::Expression diff2=dynet::square(diff);
		dynet::Expression scale_diff2=dynet::cmult(scale,diff2);
		dynet::Expression log_norm=dynet::exp(-scale_diff2);
		
		dynet::Expression phi=cutoff_function(cg,dis);
		return dynet::cmult(phi,log_norm);
	}
	
	dynet::Expression gnn_output(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& neigh_rbf,const std::vector<std::vector<unsigned>>& neigh_id);
	
	void build_neural_network();
	dynet::Expression output(dynet::ComputationGraph& cg,const dynet::Expression& x);
};
	
PLUMED_REGISTER_ACTION(GNN_AirNet,"GNN_AIRNET")

void GNN_AirNet::registerKeywords(Keywords& keys) {
	NN_GNN::registerKeywords(keys);
	
	keys.add("compulsory","RBF_SIGMA","0.32","the width of gaussian used in RBF");
	keys.add("compulsory","RBF_MIN","0.01","the width of gaussian used in RBF");
	keys.add("compulsory","INTER_NUMBER","3","the number of interaction layers of AirNet");
	keys.add("compulsory","DENSE_NUMBER","2","the number of dense blocks in the filter generator of AirNet");
	//~ keys.add("compulsory","MLP_LAYERS","3","the number of layers of MLP in feedback forward module of AirNet");
	keys.add("compulsory","PONDER_LAYERS","3","the number of hidden layers of pondering module");
	keys.add("compulsory","PONDER_DIMENSION","32","the number of hidden layers of pondering module");
	keys.add("compulsory","PONDER_ACT_FUN","RELU","the activation function in pondering module");
	keys.add("compulsory","ATTENTION_LAYERS","8","the number of layers of multi head attenaction");
	keys.add("compulsory","MAX_CYCLES","10","the maximum cycles at interaction layer.");
}

GNN_AirNet::GNN_AirNet(const ActionOptions&ao):
	NN_GNN(ao)
{
	is_self_dis=true;
	parse("INTER_NUMBER",num_inlayers);
	//~ parse("MLP_LAYERS",num_mlayers);
	parse("DENSE_NUMBER",num_dlayers);
	parse("ATTENTION_LAYERS",att_layers);
	parse("MAX_CYCLES",max_cycles);
	
	parse("PONDER_LAYERS",ponder_layers);
	
	parseVector("PONDER_DIMENSION",ponder_dim);
	if(ponder_dim.size()!=ponder_layers)
	{
		if(ponder_dim.size()==1)
		{
			unsigned dim=ponder_dim[0];
			ponder_dim.assign(ponder_layers,dim);
		}
		else
			plumed_merror("the size of PONDER_DIMENSION mismatch!");
	}
	for(unsigned i=0;i!=ponder_layers;++i)
		plumed_massert(ponder_dim[i]>0,"PONDER_DIMENSION must be larger than 0!");
	
	std::vector<std::string> str_paf;
	parseVector("PONDER_ACT_FUN",str_paf);
	if(str_paf.size()!=ponder_layers)
	{
		if(str_paf.size()==1) 
		{
			std::string af=str_paf[0];
			str_paf.assign(ponder_layers,af);
		}
		else
			plumed_merror("the size of PONDER_ACT_FUN mismatch!");
	}
	ponder_act_funs.resize(ponder_layers);
	for(unsigned i=0;i!=ponder_layers;++i)
		ponder_act_funs[i]=activation_function(str_paf[i]);
	
	plumed_massert(vec_dim%att_layers==0,"VECTOR_DIMENSION must be divisible by ATTENTION_LAYERS");
	att_dim=vec_dim/att_layers;
	weight_scale=1.0/std::sqrt(att_dim);
	
	unsigned id=0;
	for(unsigned i=0;i!=att_layers;++i)
	{
		std::vector<unsigned> ids;
		for(unsigned j=0;j!=att_dim;++j)
			ids.push_back(id++);
		row_ids.push_back(ids);
	}
	
	parse("RBF_SIGMA",rbf_sigma);
	parse("RBF_MIN",rbf_min);
	plumed_massert(rbf_min>0,"RBF_MIN must be larger than 0!");
	plumed_massert(rbf_min<cutoff,"RBF_MIN must be smaller than CUTOFF!");
	
	checkRead();
	
	log.printf("  with number of parallel attention layers for multi head attention : %d\n",int(att_layers));
	log.printf("  with dimension for each head : %d\n",int(att_layers));
	log.printf("  with log gaussian sigma at radical basis functions: %f\n",rbf_sigma);
	log.printf("  with minimual distance for radical basis functions: %f\n",rbf_min);
	log.printf("  with number of interaction layers of neural network: %d\n",int(num_inlayers));
	log.printf("  with number of dense blocks of filter generator: %d\n",int(num_dlayers));
	//~ log.printf("  with number of layers of MLP for feedback foward : %d\n",int(num_mlayers));
	log.printf("  with pondering module with %d hidden layers:\n",int(ponder_layers));
	for(unsigned i=0;i!=ponder_layers;++i)
		log.printf("    Hidden layer %d with dimension %d and activation funciton %s\n",int(i+1),int(ponder_dim[i]),str_paf[i].c_str());
}

void GNN_AirNet::build_neural_network()
{
	update_basis();
	params.resize(0);
	
	// calc_rbf
	dynet::Parameter scale=pc.add_parameters({rbf_num});
	dynet::Parameter mu=pc.add_parameters({rbf_num});
	scale.set_value(std::vector<float>(rbf_num,rbf_scale));
	mu.set_value(rbf_centers);
	params.push_back({scale,mu});
	
	// embedding
	dynet::Parameter a=pc.add_parameters({vec_dim,ntypes});
	params.push_back({a});
	
	// filter_generator
	for(unsigned idense=0;idense!=num_dlayers;++idense)
	{
		unsigned input_dim=vec_dim;
		if(idense==0)
			input_dim=rbf_num;
		dynet::Parameter dW=pc.add_parameters({vec_dim,input_dim});
		dynet::Parameter db=pc.add_parameters({vec_dim});
		params.push_back({dW,db});
	}
	
	// interaction layers
	for(unsigned l=0;l!=num_inlayers;++l)
	{
		// pondering module
		unsigned input_dim=vec_dim;
		for(unsigned p=0;p!=ponder_layers;++p)
		{
			dynet::Parameter pW=pc.add_parameters({ponder_dim[p],input_dim});
			dynet::Parameter pb=pc.add_parameters({ponder_dim[p]});
			params.push_back({pW,pb});
			input_dim=ponder_dim[p];
		}
		dynet::Parameter pW=pc.add_parameters({1,input_dim});
		dynet::Parameter pb=pc.add_parameters({1});
		params.push_back({pW,pb});
		
		// multi head attention
		dynet::Parameter Wq=pc.add_parameters({vec_dim,vec_dim});
		dynet::Parameter Wk=pc.add_parameters({vec_dim,vec_dim});
		params.push_back({Wq,Wk});
		
		// layer normalization
		dynet::Parameter G=pc.add_parameters({vec_dim});
		dynet::Parameter u=pc.add_parameters({vec_dim});
		params.push_back({G,u});
		
		//~ // feedback forward
		//~ for(unsigned iaw=0;iaw!=num_mlayers;++iaw)
		//~ {
			//~ // atom-wise
			//~ dynet::Parameter aW=pc.add_parameters({vec_dim,vec_dim});
			//~ dynet::Parameter ab=pc.add_parameters({vec_dim});
			//~ params.push_back({aW,ab});
		//~ }
	}
	
	// read out function
	input_dim=vec_dim;
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

dynet::Expression GNN_AirNet::gnn_output(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& neigh_rbf,const std::vector<std::vector<unsigned>>& neigh_id)
{
	iparm=1;
	
	// embedding layer
	unsigned iparm_begin=iparm;
	std::vector<dynet::Expression> vec_xi;
	for(unsigned i=0;i!=natoms;++i)
	{
		iparm=iparm_begin;
		dynet::Expression x0=embedding(cg,i);
		vec_xi.push_back(x0);
	}
	
	iparm_begin=iparm;
	std::vector<dynet::Expression> vgl;
	// filter_generator
	for(unsigned iatom=0;iatom!=natoms;++iatom)
	{
		iparm=iparm_begin;
		vgl.push_back(filter_generator(cg,neigh_rbf[iatom]));
	}
	
	// interaction layer
	for(unsigned l=0;l!=num_inlayers;++l)
		interaction(cg,vgl,neigh_id,vec_xi);
	dynet::Expression vxl=dynet::concatenate_cols(vec_xi);
	
	// read out function
	dynet::Expression vy=read_out(cg,vxl);
	dynet::Expression pred=dynet::sum_dim(vy,{1});
	
	return pred;
}

void GNN_AirNet::interaction(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vgl,const std::vector<std::vector<unsigned>>& neigh_id,std::vector<dynet::Expression>& vec_xi)
{
	unsigned iparm_layer=iparm;

	dynet::Expression res_weight=dynet::ones(cg,{1,natoms});
	std::vector<bool> atom_stop(natoms,false);
	
	std::vector<dynet::Expression> vxl1(natoms);
	for(unsigned ilayer=0;ilayer!=max_cycles;++ilayer)
	{
		iparm=iparm_layer;
		dynet::Expression pondering_weights=pondering(cg,vec_xi);
		dynet::Expression atom_weights=dynet::min(res_weight,pondering_weights);
		dynet::Expression weight_diff=res_weight-pondering_weights;
		
		res_weight=dynet::max(weight_diff,dynet::zeros(cg,{1,natoms}));
		std::vector<float> value_diff=dynet::as_vector(weight_diff.value());
		std::vector<float> value_weights=dynet::as_vector(atom_weights.value());

		bool stop_cycle=true;
		//~ std::cout<<"cycles "<<ilayer<<":";
		unsigned iparm_atom=iparm;
		for(unsigned iatom=0;iatom!=natoms;++iatom)
		{
			if(atom_stop[iatom]||value_weights[iatom]<=0)
			{
				vxl1[iatom]=vec_xi[iatom];
			}
			else
			{
				//~ std::cout<<" "<<iatom<<"("<<value_weights[iatom]<<"),";
				iparm=iparm_atom;
				std::vector<dynet::Expression> vec_xij;
				for(unsigned j=0;j!=neigh_id[iatom].size();++j)
					vec_xij.push_back(vec_xi[neigh_id[iatom][j]]);
				dynet::Expression xij=concatenate_cols(vec_xij);
				
				dynet::Expression qi;
				dynet::Expression Ki;
				dynet::Expression Vi;
				
				positional_embedding(cg,vec_xi[iatom],xij,vgl[iatom],qi,Ki,Vi);
				
				std::vector<unsigned> id={iatom};
				dynet::Expression dxi=multi_head_attention(cg,qi,Ki,Vi);
				dynet::Expression ww=select_cols(atom_weights,id);
				dynet::Expression al=vec_xi[iatom]+dxi*ww;
				
				vxl1[iatom]=layer_normalization(cg,al);
			}

			if(value_diff[iatom]<=0)
				atom_stop[iatom]=true;
			else
				stop_cycle=false;
		}
		vec_xi.swap(vxl1);
		//~ std::cout<<std::endl;
		
		if(stop_cycle)
			break;
	}
}

dynet::Expression GNN_AirNet::pondering(dynet::ComputationGraph& cg,const std::vector<dynet::Expression>& vec_xi)
{
	dynet::Expression vx = dynet::concatenate_cols(vec_xi);
	
	for(unsigned p=0;p!=ponder_layers;++p)
	{
		dynet::Expression pW = dynet::parameter(cg, params[iparm][0]);
		dynet::Expression pb = dynet::parameter(cg, params[iparm][1]);
		++iparm;
		
		vx = dy_act_fun(pW*vx+pb,ponder_act_funs[p]);
	}
	
	dynet::Expression pW = dynet::parameter(cg, params[iparm][0]);
	dynet::Expression pb = dynet::parameter(cg, params[iparm][1]);
	++iparm;

	// Sigmoid
	return dynet::logistic(pW*vx+pb);
}

void GNN_AirNet::positional_embedding(dynet::ComputationGraph& cg,const dynet::Expression& xi,const dynet::Expression& xij,const dynet::Expression& gij,dynet::Expression& qi,dynet::Expression& Ki,dynet::Expression& Vi)
{
	dynet::Expression Wq = dynet::parameter(cg, params[iparm][0]);
	dynet::Expression Wk = dynet::parameter(cg, params[iparm][1]);
	++iparm;
	
	qi=Wq*xi;
	Ki=Wk*(xij+gij);
	Vi=xij;
}

dynet::Expression GNN_AirNet::multi_head_attention(dynet::ComputationGraph& cg,const dynet::Expression& qi,const dynet::Expression& Ki,const dynet::Expression& Vi)
{
	std::vector<dynet::Expression> vdxi;
	for(unsigned i=0;i!=att_layers;++i)
	{
		dynet::Expression qi_mh=dynet::select_rows(qi,row_ids[i]);
		dynet::Expression Ki_mh=dynet::select_rows(Ki,row_ids[i]);
		dynet::Expression Vi_mh=dynet::select_rows(Vi,row_ids[i]);
		
		dynet::Expression cij_mh=dynet::softmax(dynet::transpose(Ki_mh)*qi_mh*weight_scale);
		
		vdxi.push_back(Vi_mh*cij_mh);
	}
	return dynet::concatenate(vdxi);
}

dynet::Expression GNN_AirNet::output(dynet::ComputationGraph& cg,const dynet::Expression& x)
{
	std::vector<std::vector<dynet::Expression>> mat_dis(natoms);
	std::vector<dynet::Expression> vec_rbf(natoms);
	std::vector<std::vector<unsigned>> neigh_id(natoms);
	std::vector<dynet::Expression> acoord;
	
	for(unsigned i=0;i!=natoms;++i)
	{
		std::vector<unsigned> id={i};
		acoord.push_back(dynet::select_cols(x,id));
	}

	for(unsigned i=0;i!=natoms;++i)
	{
		for(unsigned j=i+1;j<natoms;++j)
		{
			dynet::Expression dis=dynet::sqrt(squared_distance(
				acoord[i],acoord[j]));

			std::vector<float> vdis=dynet::as_vector(dis.value());
			if(vdis[0]<cutoff)
			{
				mat_dis[i].push_back(dis);
				mat_dis[j].push_back(dis);
				neigh_id[i].push_back(j);
				neigh_id[j].push_back(i);
			}
		}
		dynet::Expression cdis=dynet::concatenate_cols(mat_dis[i]);
		vec_rbf[i]=calc_rbf(cg,cdis);
	}
	
	return gnn_output(cg,vec_rbf,neigh_id);
}

}
}

#endif
