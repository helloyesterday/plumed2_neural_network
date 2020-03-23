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
#include "Function.h"
#include "ActionRegister.h"
#include "tools/Matrix.h"
#include "tools/IFile.h"

#include <cmath>

using namespace std;

namespace PLMD {
namespace function {

//+PLUMEDOC FUNCTION NEURAL_NETWORK
/*

*/
//+ENDPLUMEDOC


class neural_network :	public Function
{
	unsigned narg;
	unsigned nlayer;
	unsigned output_dim;
	std::string coe_file;
	std::string reg_file;
	std::vector<double> input;
	//~ std::vector<std::vector<double> > layer_input;
	std::vector<std::vector<double> > layer_output;
	std::vector<unsigned> layer_dim;
	std::vector<unsigned> layer_col;
	std::vector<Matrix<double> > weight_matrix;
	std::vector<Matrix<double> > layer_deriv;
	std::vector<vector<double> > offset_vector;
	std::vector<std::string> act_funs;
	std::vector<Value*> valueOut;
	
	const std::set<std::string> afuns={"none","sigmoid","ReLU","tanh","softmax"};
public:
	explicit neural_network(const ActionOptions&);
	void calculate();
	static void registerKeywords(Keywords& keys);
private:
	inline double sigmoid(double value,double& deriv);
	inline double ReLU(double value,double& deriv);
	inline double hyper_tan(double value,double& deriv);
	std::vector<double> softmax(const std::vector<double>& input_vec,Matrix<double>& deriv_mat);
};


PLUMED_REGISTER_ACTION(neural_network,"NEURAL_NETWORK")

void neural_network::registerKeywords(Keywords& keys) {
	Function::registerKeywords(keys);
	useCustomisableComponents(keys);
	keys.use("ARG");
	keys.add("compulsory","PARM_FILE","read the file of parameters");
	keys.add("compulsory","REG_PARM","reg_parm.data","the regular form of parameters");
}

neural_network::neural_network(const ActionOptions&ao):
Action(ao),
Function(ao),
narg(getNumberOfArguments()),
nlayer(0),
input(getNumberOfArguments())
{
	parse("PARM_FILE",coe_file);
	parse("REG_PARM",reg_file);
	IFile iparm;
    iparm.link(*this);

	if(!iparm.FileExist(coe_file))
		plumed_merror("Cannot find parameter file " + coe_file );

    iparm.open(coe_file);
    
	iparm.allowIgnoredFields();
	
	int ival;
	if(iparm.FieldExist("ARGUMENT_NUMBER"))
		iparm.scanField("ARGUMENT_NUMBER",ival);
	else
		plumed_merror("Could not find parameter ARGUMENT_NUMBER in file: "+coe_file);
	
	plumed_massert(unsigned(ival)==narg,"the number of arguments mismatch!");
	
	if(iparm.FieldExist("LAYER_NUMBER"))
		iparm.scanField("LAYER_NUMBER",ival);
	else
		plumed_merror("Could not find parameter LAYER_NUMBER in file: "+coe_file);
	nlayer=ival;
	plumed_massert(nlayer>0,"the LAYER_NUMBER must larger than 0!");
	
	layer_dim.resize(nlayer);
	layer_col.resize(nlayer);
	act_funs.resize(nlayer);
	
	layer_col[0]=narg;
	for(unsigned i=0;i!=nlayer;++i)
	{
		std::string id;
		Tools::convert(i,id);
		iparm.scanField("LAYER_DIM_"+id,ival);
		layer_dim[i]=ival;
		if(i+1<nlayer)
			layer_col[i+1]=layer_dim[i];
	}
	output_dim=layer_dim.back();
	
	for(unsigned i=0;i!=nlayer;++i)
	{
		std::string id;
		Tools::convert(i,id);
		iparm.scanField("LAYER_ACTION_"+id,act_funs[i]);
		if(afuns.count(act_funs[i])==0)
			plumed_merror("unrecognized activation functions: "+act_funs[i]);
	}

	unsigned pre_dim=narg;
	for(unsigned i=0;i!=nlayer;++i)
	{
		Matrix<double> wm(layer_dim[i],pre_dim);
		std::vector<double> bv(layer_dim[i],0);
		weight_matrix.push_back(wm);
		layer_deriv.push_back(wm);
		offset_vector.push_back(bv);
		//~ layer_input.push_back(bv);
		layer_output.push_back(bv);
		pre_dim=layer_dim[i];
	}
	unsigned lid;
	int ilid;
	while(iparm.scanField("layer_id",ilid))
	{
		lid=ilid;
		plumed_massert(lid<nlayer,"the layer id should be less than LAYER_NUMBER");
		unsigned row,col;
		double val;
		iparm.scanField("layer_row",ival);
		row=ival;
		plumed_massert(row<layer_dim[lid],"the layer row should be less than LAYER_DIM");
		iparm.scanField("layer_col",ival);
		col=ival;
		iparm.scanField("parameter",val);
		
		if(col<layer_col[lid])
			weight_matrix[lid][row][col]=val;
		else if(col==layer_col[lid])
			offset_vector[lid][row]=val;
		else
			plumed_merror("the layer column should be less than LAYER_COL");
		
		iparm.scanField();
	}
	iparm.close();

	if(output_dim==1)
	{
		addValueWithDerivatives();
		setNotPeriodic();
	}
	else
	{
		valueOut.resize(output_dim);
		for(unsigned i=0;i!=output_dim;++i)
		{
			std::string id;
			Tools::convert(i,id);
			std::string conp="out["+id+"]";
			addComponentWithDerivatives(conp);
			componentIsNotPeriodic(conp);
			valueOut[i]=getPntrToComponent(conp);
		}
	}
	
	checkRead();
	
	OFile oreg;
	oreg.link(*this);
	oreg.open(reg_file);
	oreg.addConstantField("LAYER_ID");
	oreg.addConstantField("LAYER_DIM");
	oreg.addConstantField("LAYER_COLUMN");
	oreg.addConstantField("ACTIVATION");
	oreg.fmtField(" %f");
	
	for(unsigned i=0;i!=nlayer;++i)
	{
		oreg.printField("LAYER_ID",int(i));
		oreg.printField("LAYER_DIM",int(layer_dim[i])); 
		oreg.printField("LAYER_COLUMN",int(layer_col[i]));
		oreg.printField("ACTIVATION",act_funs[i].c_str());
		
		for(unsigned m=0;m!=weight_matrix[i].nrows();++m)
		{
			std::string rowid;
			Tools::convert(m,rowid);
			std::string srow="ROW_"+rowid;
			oreg.printField("MATRIX",srow);
			for(unsigned n=0;n!=weight_matrix[i].ncols();++n)
			{
				std::string colid;
				Tools::convert(n,colid);
				std::string scol="COL_"+colid;
				oreg.printField(scol,weight_matrix[i][m][n]);
			}
			oreg.printField("BIAS_VECTOR",offset_vector[i][m]);
			oreg.printField();
		}
		oreg.flush();
	}
	oreg.close();

	log.printf("  with number of input argument: %d\n",int(narg));
	log.printf("  with number of output argument: %d\n",int(output_dim));
	log.printf("  with number of layer: %d\n",int(nlayer));
	for(unsigned i=0;i!=nlayer;++i)
		log.printf("    Layer %d with dimension %d and activation funciton %s\n",int(i),int(layer_dim[i]),act_funs[i].c_str());
	log.printf("  with regular parameter file: %s\n",reg_file.c_str());
}

void neural_network::calculate()
{
	
	for(unsigned i=0;i!=narg;++i)
		input[i]=getArgument(i);
	std::vector<double> layer_input=input;
	std::vector<Matrix<double> > layer_deriv(weight_matrix);
	for(unsigned i=0;i!=nlayer;++i)
	{
		std::vector<double> zz;
		mult(weight_matrix[i],layer_input,zz);
		if(act_funs[i]=="none")
		{
			for(unsigned m=0;m!=layer_dim[i];++m)
				layer_output[i][m]=zz[m]+offset_vector[i][m];
		}
		else if(act_funs[i]=="softmax")
		{
			for(unsigned m=0;m!=layer_dim[i];++m)
				zz[m]+=offset_vector[i][m];
			Matrix<double> dev_mat;
			layer_output[i]=softmax(zz,dev_mat);
			mult(dev_mat,weight_matrix[i],layer_deriv[i]);
		}
		else
		{
			for(unsigned m=0;m!=layer_dim[i];++m)
			{
				zz[m]+=offset_vector[i][m];
				double dev=1.0;
				
				if(act_funs[i]=="sigmoid")
					layer_output[i][m]=sigmoid(zz[m],dev);
				else if(act_funs[i]=="ReLU")
					layer_output[i][m]=ReLU(zz[m],dev);
				else if(act_funs[i]=="tanh")
					layer_output[i][m]=hyper_tan(zz[m],dev);
					
				for(unsigned n=0;n!=layer_col[i];++n)
					layer_deriv[i][m][n]*=dev;
			}
		}
		layer_input=layer_output[i];
	}
	std::vector<double> output=layer_output.back();
	Matrix<double> output_deriv;
	Matrix<double> input_deriv=layer_deriv.back();
	for(unsigned i=1;i!=nlayer;++i)
	{
		mult(input_deriv,layer_deriv[nlayer-i-1],output_deriv);
		input_deriv=output_deriv;
	}
	
	if(output_dim==1)
	{
		setValue(output[0]);
		for(unsigned j=0;j!=narg;++j)
			setDerivative(j,output_deriv[0][j]);
	}
	else
	{
		for(unsigned i=0;i!=output_dim;++i)
		{
			valueOut[i]->set(output[i]);
			for(unsigned j=0;j!=narg;++j)
				setDerivative(valueOut[i],j,output_deriv[i][j]);
		}
	}
}

inline double neural_network::sigmoid(double value,double& deriv)
{
	double act=1.0/(1.0+exp(-value));
	deriv=act*(1.0-act);
	return act;
}

inline double neural_network::ReLU(double value,double& deriv)
{
	if(value>0)
	{
		deriv=1;
		return value;
	}
	else
	{
		deriv=0;
		return 0;
	}
}

inline double neural_network::hyper_tan(double value,double& deriv)
{
	double act=tanh(value);
	deriv=1.0-act*act;
	return act;
}

std::vector<double> neural_network::softmax(const std::vector<double>& input_vec,Matrix<double>& deriv_mat)
{
	std::vector<double> act_vec(input_vec);
	deriv_mat.resize(input_vec.size(),input_vec.size());
	double sum=0;
	for(unsigned i=0;i!=act_vec.size();++i)
		sum+=act_vec[i];
	for(unsigned i=0;i!=act_vec.size();++i)
		act_vec[i]/=sum;
	for(unsigned i=0;i!=act_vec.size();++i)
	{
		for(unsigned j=0;j!=act_vec.size();++j)
		{
			if(i==j)
				deriv_mat[i][j]=act_vec[i]*(1.0-act_vec[j]);
			else
				deriv_mat[i][j]=act_vec[i]*act_vec[j];
		}
	}
	return act_vec;
}


}
}


