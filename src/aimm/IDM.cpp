/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_DYNET

#include <numeric>
#include "vesselbase/ActionWithMultiAveraging.h"
#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "analysis/AverageVessel.h"
#include "tools/IFile.h"
#include "tools/Matrix.h"
#include "reference/ReferenceConfiguration.h"
#include "reference/ReferenceValuePack.h"

//+PLUMEDOC DIMRED IDM 
/* 
Harmonic linear discriminant analysis (IDM) using a large number of collective variables as input.

The theory of IDM can be found at paper:
Mendels, D, Piccini, GM and Parrinello, M. J. Phys. Chem. Lett. 9, 2776 (2018) https://doi.org/10.1021/acs.jpclett.8b00733

\par Examples

All the collective variables in PLUMED2 can be used as the basis-set. A simple example is as below,
three CVs: cv1, cv2 and cv3 are used as the basis-set of IDM. There are the trajectories files colvar.1.data
and colvar.2.data, which respond to two different states. The program with output every 5000 steps:

\verbatim
s1cv1: READ FILE=colvar.1.data VALUES=cv1 IGNORE_TIME
s1cv2: READ FILE=colvar.1.data VALUES=cv2 IGNORE_TIME
s1cv3: READ FILE=colvar.1.data VALUES=cv3 IGNORE_TIME

s2cv1: READ FILE=colvar.2.data VALUES=cv1 IGNORE_TIME
s2cv2: READ FILE=colvar.2.data VALUES=cv2 IGNORE_TIME
s2cv3: READ FILE=colvar.2.data VALUES=cv3 IGNORE_TIME

IDM ...
 NUM=2
 ARG1=s1cv1,s1cv2,s1cv3
 ARG2=s2cv1,s2cv2,s2cv3
 ARG_NUM=3
 OUTPUT_STRIDE=5000
... IDM
\endverbatim

You can also use linear discriminant analysis (LDA):

\verbatim
s1cv1: READ FILE=colvar.1.data VALUES=cv1 IGNORE_TIME
s1cv2: READ FILE=colvar.1.data VALUES=cv2 IGNORE_TIME
s1cv3: READ FILE=colvar.1.data VALUES=cv3 IGNORE_TIME

s2cv1: READ FILE=colvar.2.data VALUES=cv1 IGNORE_TIME
s2cv2: READ FILE=colvar.2.data VALUES=cv2 IGNORE_TIME
s2cv3: READ FILE=colvar.2.data VALUES=cv3 IGNORE_TIME

IDM ...
 USE_LDA
 NUM=2
 ARG1=s1cv1,s1cv2,s1cv3
 ARG2=s2cv1,s2cv2,s2cv3
 ARG_NUM=3
 OUTPUT_STRIDE=5000
... IDM
\endverbatim

*/
//+ENDPLUMEDOC

namespace PLMD {
namespace analysis {

class IDM : public vesselbase::ActionWithMultiAveraging {
private:
	// If we are reusing data are we ignoring the reweighting in that data
	unsigned tot_nargs;
	unsigned output_steps;
	unsigned _narg;
	unsigned idata;
	unsigned irecord;
	unsigned ncomp;
	
	bool use_lda;
	
	std::string avg_file;
	std::string cov_file;
	std::string mat_file;
	std::string eigval_file;
	std::string eigvec_file;
	
	std::vector<std::vector<double> > mcov;
	std::vector<std::vector<double> > mavg;
	std::vector<std::vector<std::vector<double> > > record_cov;
	std::vector<std::vector<std::vector<double> > > record_avg;
	std::vector<unsigned> record_steps;
	std::vector<double> cw0;
	std::vector<double> cnorm;
	
	void calc_avg(bool finished);
public:
	static void registerKeywords( Keywords& keys );
	explicit IDM( const ActionOptions& );
	~IDM();
	void calculate(){}
	void apply(){}
	void performOperations( const bool& from_update );
	void performTask( const unsigned& , const unsigned& , MultiValue& ) const { plumed_error(); }
	
	void accumulate();
	void performAnalysis();
	void runFinalJobs();
	void runAnalysis();
	bool isPeriodic(){ plumed_error(); return false; }
};

PLUMED_REGISTER_ACTION(IDM,"IDM")

void IDM::registerKeywords( Keywords& keys ){
	vesselbase::ActionWithMultiAveraging::registerKeywords( keys );
	keys.remove("SERIAL"); keys.remove("LOWMEM"); 
	keys.remove("NORMALIZATION");
	keys.add("compulsory","ARG_NUM","how many arguments for each states");
	keys.add("compulsory","OUTPUT_STRIDE","0","the frequenct to output. 0 means only output the final result");
	keys.add("compulsory","AVERAGE_FILE","hlda_average","the file to output the average");
	keys.add("compulsory","CORRELATION_FILE","hlda_correlation","the file to output the coorelation");
	keys.add("compulsory","IDM_MATRIX_FILE","hlda_matrix.data","the file to output the final matrix");
	keys.add("compulsory","EIGENVECTOR_FILE","hlda_eigenvector","the file to output the result");
	keys.add("compulsory","EIGENVALUE_FILE","hlda_eigenvalue.data","the file to output the eigen value");
	keys.addFlag("USE_LDA",false,"use LDA instead of IDM");
	keys.add("optional","EIGEN_NUMBERS","how many eigenvectors to be output (from large to small)");
}

IDM::IDM(const ActionOptions&ao):
Action(ao),
ActionWithMultiAveraging(ao),
idata(0),
irecord(0)
{
	addValue(); // Create a value so that we can output the average
	tot_nargs=getNumberOfArguments();
	setNotPeriodic();
	
	parse("ARG_NUM",_narg);
	
	for(unsigned i=0;i!=getNumberOfStates();++i)
	{
		if(getNumberOfArguments(i)!=_narg)
		{
			std::string stid,str_dim,str_narg;
			Tools::convert(i,stid);
			Tools::convert(_narg,str_dim);
			Tools::convert(getNumberOfArguments(i),str_narg);
			std::string err_msg="The number of arguments ("+str_narg+") at state "+stid+" is not equal to the default number ("+str_dim+")\n";
			plumed_merror(err_msg);
		}
	}
	
	cw0.assign(_narg,1.0);
	cnorm.assign(_narg,0);
	
	mavg.assign(getNumberOfStates(),std::vector<double>(_narg,0));
	mcov.assign(getNumberOfStates(),std::vector<double>(_narg*(_narg+1)/2,0));
	
	parse("OUTPUT_STRIDE",output_steps);
	parse("AVERAGE_FILE",avg_file);

	parse("CORRELATION_FILE",cov_file);
	parse("IDM_MATRIX_FILE",mat_file);
	parse("EIGENVECTOR_FILE",eigvec_file);
	parse("EIGENVALUE_FILE",eigval_file);
	
	ncomp=_narg;
	parse("EIGEN_NUMBERS",ncomp);
	plumed_massert(ncomp>0,"the EIGEN_NUMBER must be larger than 0!");
	plumed_massert(ncomp<=_narg,"the EIGEN_NUMBER cannot be larger than the number of CVs!");
	
	parseFlag("USE_LDA",use_lda);

	checkRead();
	
	log.printf("  with number of states:%d\n",getNumberOfStates());
	log.printf("  with number of argument for each state:%d\n",getNumberOfStates());
	log.printf("  with output stride: %d.\n",output_steps);
	log.printf("  with average output file: %s\n",avg_file.c_str());
	log.printf("  with correlation output file: %s\n",cov_file.c_str());
	log<<"  Bibliography "<<plumed.cite("Mendels, Piccini and Parrinello, J. Phys. Chem. Lett. 9, 2776 (2018)");
	log<<"\n";
}

IDM::~IDM()
{
}

void IDM::performAnalysis()
{
	if(output_steps==0||idata>record_steps.back())
	{
		calc_avg(true);
	}
	else
	{
		mavg=record_avg.back();
		mcov=record_cov.back();
	}
	
	log.printf("Fininished reading.\n");
	log.printf("  with reading steps: %d.\n",idata);
	log.printf("  with output steps: %d.\n",irecord);
	
	for(unsigned is=0;is!=getNumberOfStates();++is)
	{
		std::string id;
		Tools::convert(is,id);
		
		OFile oavg;
		oavg.link(*this);
		std::string avg_name=avg_file+id+".data";
		oavg.open(avg_name.c_str());
		oavg.fmtField(" %e");
		
		OFile ocov;
		ocov.link(*this);
		std::string cov_name=cov_file+id+".data";
		ocov.open(cov_name.c_str());
		ocov.fmtField(" %e");
		
		for(unsigned ir=0;ir!=irecord;++ir)
		{
			oavg.printField("step",int(record_steps[ir]));
			ocov.printField("step",int(record_steps[ir]));
			unsigned cid=0;
			
			for(unsigned i=0;i!=_narg;++i)
			{
				std::string iid;
				Tools::convert(i,iid);
				std::string apf="avg_"+getLabelOfArgument(is,i);
				oavg.printField(apf,record_avg[ir][is][i]);
				for(unsigned j=0;j!=i+1;++j)
				{
					std::string jid;
					Tools::convert(j,jid);
					std::string cpf="i"+iid+"j"+jid;
					ocov.printField(cpf,record_cov[ir][is][cid++]);
				}
			}
			oavg.printField();
			ocov.printField();
		}
		oavg.close();
		ocov.close();
	}
	
	OFile oeigval;
	oeigval.link(*this);
	oeigval.open(eigval_file.c_str());
	std::vector<std::string> col_args;
	std::vector<std::string> row_args;
	for(unsigned i=0;i!=_narg;++i)
	{
		std::string id;
		Tools::convert(i,id);
		std::string colid="COL"+id;
		std::string rowid="ROW"+id;
		col_args.push_back(colid);
		row_args.push_back(rowid);
	}
	OFile omat;
	omat.link(*this);
	omat.open(mat_file.c_str());
	omat.fmtField(" %e");
	omat.addConstantField("DIMENSION");
	omat.printField("DIMENSION",int(_narg));
	omat.addConstantField("NPOINTS");
	omat.printField("NPOINTS",int(irecord));
	omat.addConstantField("STEP");
	
	std::vector<double> max_value(_narg,0);
	std::vector<unsigned> max_ids(_narg,0);
	std::vector<std::vector<std::vector<double> > > eigvec_points;
	for(unsigned ir=0;ir!=irecord;++ir)
	{
		log.printf("  Setp: %d\n",int(record_steps[ir]));
		std::vector<std::vector<double> > mux;
		//~ vec mu_avg(_narg,fill::zeros);
		std::vector<double> mu_avg(_narg,0);
		
		//~ mat cov_inv(_narg,_narg,fill::zeros);
		Matrix<double> cov_inv(_narg,_narg);
		Matrix<double> within_class(_narg,_narg);
		for(unsigned i=0;i!=_narg;++i)
		{
			for(unsigned j=0;j!=_narg;++j)
			{
				cov_inv[i][j]=0;
				within_class[i][j]=0;
			}
		}
		
		for(unsigned is=0;is!=getNumberOfStates();++is)
		{
			const std::vector<double>& mu0=record_avg[ir][is];
			const std::vector<double>& cov0=record_cov[ir][is];
			mux.push_back(mu0);
			
			unsigned cid=0;
			Matrix<double> mc(_narg,_narg);
			for(unsigned i=0;i!=_narg;++i)
			{
				mu_avg[i]+=mu0[i];
				for(unsigned j=0;j!=i+1;++j)
				{
					double vcov=cov0[cid++];
					mc[i][j]=vcov;
					if(i!=j)
						mc[j][i]=vcov;
				}
			}
			Matrix<double> imc(_narg,_narg);
			if(use_lda)
				imc=mc;
			else
				Invert(mc,imc);

			for(unsigned i=0;i!=_narg;++i)
			{
				for(unsigned j=0;j!=_narg;++j)
					cov_inv[i][j]+=imc[i][j];
			}
		}
		for(unsigned i=0;i!=_narg;++i)
			mu_avg[i]/=getNumberOfStates();
		
		if(use_lda)
		{
			Matrix<double> tmp_cov;
			Invert(cov_inv,tmp_cov);
			cov_inv=tmp_cov;
		}
		
		for(unsigned is=0;is!=getNumberOfStates();++is)
		{
			std::vector<double> mu0m(mux[is]);
			for(unsigned i=0;i!=_narg;++i)
				mu0m[i]-=mu_avg[i];

			for(unsigned i=0;i!=_narg;++i)
			{
				for(unsigned j=0;j!=_narg;++j)
					within_class[i][j]+=mu0m[i]*mu0m[j];
			}
		}

		Matrix<double> tot_class;
		mult(cov_inv,within_class,tot_class);

		omat.printField("STEP",int(record_steps[ir]));
		for(unsigned i=0;i!=_narg;++i)
		{
			omat.printField("MATRIX",row_args[i]);
			for(unsigned j=0;j!=_narg;++j)
				omat.printField(col_args[j],cov_inv[i][j]);
			omat.printField();
		}
		omat.flush();
			omat.printField("STEP",int(record_steps[ir]));
		for(unsigned i=0;i!=_narg;++i)
		{
			omat.printField("MATRIX",row_args[i]);
			for(unsigned j=0;j!=_narg;++j)
				omat.printField(col_args[j],within_class[i][j]);
			omat.printField();
		}
		omat.flush();
		
		std::vector<double> wtr,wti;
		Matrix<double> lvtr,lvti,rvtr,rvti;
		
		//~ eig_gen(wt,vt,tot_class);
		diagGenMat(tot_class,wtr,wti,lvtr,lvti,rvtr,rvti);
		
		std::multimap<double,std::vector<double> > eigs0;

		for(unsigned i=0;i!=_narg;++i)
		{
			std::vector<double> rowvec;
			for(unsigned j=0;j!=_narg;++j)
				rowvec.push_back(rvtr[i][j]);
			eigs0.insert(std::pair<double,std::vector<double> >(wtr[i],rowvec));
		}

		std::vector<double> eigval0;
		std::vector<std::vector<double> > eigvec0;
		
		for(std::multimap<double,std::vector<double> >::reverse_iterator i=eigs0.rbegin();i!=eigs0.rend();++i)
		{
			eigval0.push_back(i->first);
			eigvec0.push_back(i->second);
		}
		
		for(unsigned i=0;i!=_narg;++i)
		{
			double max=0;
			unsigned max_id=_narg;
			for(unsigned j=0;j!=_narg;++j)
			{
				if(fabs(eigvec0[i][j])>max)
				{
					max=fabs(eigvec0[i][j]);
					max_id=j;
				}
			}
			max_ids[i]=max_id;
			if(ir>0&&(eigvec0[i][max_id]*eigvec_points.back()[i][max_id]<0))
			{
				for(unsigned j=0;j!=_narg;++j)
					eigvec0[i][j]*=-1;
			}
			max_value[i]=eigvec0[i][max_id];
		}
		
		eigvec_points.push_back(eigvec0);
		
		log.printf("  Eigenvalues:");
		for(unsigned i=0;i!=_narg;++i)
			log.printf(" %e",eigval0[i]);
		log.printf("\n");
		
		oeigval.fmtField(" %e");
		oeigval.printField("STEP",int(record_steps[ir]));
		for(unsigned i=0;i!=_narg;++i)
		{
			std::string id;
			Tools::convert(i,id);
			std::string eigid="EIGVAL"+id;
			oeigval.printField(eigid,eigval0[i]);
		}
		oeigval.printField();
		oeigval.flush();
	}
	oeigval.close();
	omat.close();
	
	for(unsigned i=0;i!=ncomp;++i)
	{
		OFile oeigvec;
		oeigvec.link(*this);
		std::string id;
		Tools::convert(i,id);
		std::string name=eigvec_file+id+".data";
		oeigvec.open(name.c_str());
		oeigvec.addConstantField("COMPONENT");
		oeigvec.printField("COMPONENT",int(i));
		oeigvec.fmtField(" %e");
		for(unsigned j=0;j!=eigvec_points.size();++j)
		{
			oeigvec.printField("STEP",int(record_steps[j]));
			for(unsigned k=0;k!=_narg;++k)
			{
				double vvec=eigvec_points[j][i][k];
				if(max_value[i]<0)
					vvec*=-1;
				std::string id;
				Tools::convert(k,id);
				std::string eigid="EIGVEC"+id;
				oeigvec.printField(eigid,vvec);
			}
			oeigvec.printField();
		}
		oeigvec.flush();
		oeigvec.close();
	}
}

void IDM::accumulate(){
	
	if(UseWeights())
		cw0=cweights;
	
	for(unsigned is=0;is!=getNumberOfStates();++is)
	{
		std::vector<double> data0(_narg);
		//~ log.printf("\t%d\t%d",int(idata),int(is));
		for(unsigned i=0;i!=_narg;++i)
		{
			double arg0=getArgument(is,i);
			//~ log.printf("\t%f",arg0);
			data0[i]=arg0;
			mavg[is][i]+=arg0*cw0[is];
		}
		unsigned id=0;
		for(unsigned i=0;i!=_narg;++i)
		{
			for(unsigned j=0;j!=i+1;++j)
				mcov[is][id++]+=data0[i]*data0[j]*cw0[is];
		}
		//~ log.printf("\t%f\n",cw0[is]);
		cnorm[is]+=cw0[is];
	}
	// Increment data counter
	++idata;
	
	if(output_steps!=0&&idata%output_steps==0)
		calc_avg(false);
}

void IDM::calc_avg(bool finished)
{
	std::vector<std::vector<double> > doavg(mavg);
	std::vector<std::vector<double> > docov(mcov);
	
	for(unsigned is=0;is!=getNumberOfStates();++is)
	{
		unsigned id=0;
		for(unsigned i=0;i!=_narg;++i)
		{
			doavg[is][i]/=cnorm[is];
			for(unsigned j=0;j!=i+1;++j)
				docov[is][id++]/=cnorm[is];
		}
	}
	for(unsigned is=0;is!=getNumberOfStates();++is)
	{
		unsigned id=0;
		for(unsigned i=0;i!=_narg;++i)
		{
			for(unsigned j=0;j!=i+1;++j)
				docov[is][id++]-=doavg[is][i]*doavg[is][j];
		}
	}

	if(finished)
	{
		mavg=doavg;
		mcov=docov;
	}
	
	record_avg.push_back(doavg);
	record_cov.push_back(docov);
	record_steps.push_back(idata);
	++irecord;
}


void IDM::performOperations( const bool& from_update ){
  accumulate();
}

void IDM::runAnalysis(){
	performAnalysis(); idata=0;
}

void IDM::runFinalJobs() {
  runAnalysis(); 
}

}
}

#endif
