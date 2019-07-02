/*!
 * 
 *
 * \brief       Example of an experiment using the CMA-ES on several benchmark functions
 *
 * \author      O.Krause
 * \date        2014
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
// Implementations of the xNES
#include "fCMA.h"
#include "fCMA_NNH.h"
#include "pcCMSA.h"
#include "pcfCMA.h"
#include "CSA.h"
//benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>

//misc
#include <fstream>
#include <boost/filesystem.hpp>
using namespace shark;

struct NoisySphere : public SingleObjectiveFunction {
	
	NoisySphere(std::size_t numberOfVariables):m_numberOfVariables(numberOfVariables){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisySphere"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::gauss(random::globalRng(), 0,1);
		}
		return x;
	}

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		
		double var = 0.0001;
		return norm_sqr(x)+random::gauss(random::globalRng(),0.0, var);
	}
private:
	std::size_t m_numberOfVariables;
};


struct NoisyEllipsoid : public benchmarks::Ellipsoid{
	
	NoisyEllipsoid(std::size_t numberOfVariables):benchmarks::Ellipsoid(numberOfVariables, 0.01){
		m_features |= IS_NOISY;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisyEllipsoid"; }

	double eval(SearchPointType const& x) const {
		double var = 0.0001;
		return benchmarks::Ellipsoid::eval(x)+random::gauss(random::globalRng(),0.0, var);
	}
};

typedef boost::shared_ptr<SingleObjectiveFunction > Function;
template<class Optimizer>
void evaluate(std::vector<Function >const& functions, std::vector<Function >const& functionsEval, std::size_t m){
	Optimizer cma;
	for(std::size_t f = 0; f != functions.size(); ++f){
		//init function
		random::globalRng().seed(142+f);
		functions[f]->init();
		std::size_t d = functions[f]->numberOfVariables();
		unsigned int lambda = Optimizer::suggestLambda(d);
		std::size_t budget = Optimizer::suggestLambda(1000)*m*d+1;
		//initialize the optimizer
		cma.init( *functions[f] );
		
		std::string directory ="./results/"+cma.name()+"/";
		boost::filesystem::create_directory(directory);
		std::string fileName=directory+functions[f]->name() + "-" + std::to_string(d)+".txt";
		std::ofstream str(fileName.c_str());
		str<<"#iteration value sigma \n"; 
		std::cout<<functions[f]->name()<<" "<<d<<std::endl;
		//optimize
		std::size_t lastPrint = 0;
		while(functions[f]->evaluationCounter() < budget){
			str.precision( 7 );
			str<<functions[f]->evaluationCounter()<<" "<<(*functionsEval[f])(cma.solution().point)<<" "<<cma.sigma()<<" "<<cma.lambda()<<"\n";
			cma.step(*functions[f]);
			if(functions[f]->evaluationCounter() - lastPrint >= 1000*lambda){
				std::cout<<functions[f]->evaluationCounter()<<" "<<(*functionsEval[f])(cma.solution().point)<<" "<<cma.sigma()<<" "<<cma.lambda()<<std::endl;
				lastPrint = functions[f]->evaluationCounter();
			}
		}
		std::cout<<functions[f]->evaluationCounter()<<" "<<(*functionsEval[f])(cma.solution().point)<<" "<<cma.sigma()<<" "<<cma.lambda()<<std::endl;
	}
}

int main( int argc, char ** argv ) {
	using namespace shark::benchmarks;
	boost::filesystem::create_directory("./results");
	{
		std::size_t dims=1000;
		std::size_t epochs=10;
		Function sphere = Function(new Sphere(dims));
		Function elli = Function(new Ellipsoid(dims, 1.e-2));
		std::vector<Function > functions;
		functions.push_back(sphere);
		functions.push_back(elli);
		evaluate<fCMA>(functions, functions, epochs);
		evaluate<pcCMSA>(functions, functions, epochs);
		evaluate<CSA>(functions, functions, epochs);
	}
	
	{
		std::size_t dims=8000;
		std::size_t epochs=10;
		Function sphere = Function(new Sphere(dims));
		Function elli = Function(new Ellipsoid(dims, 1.e-2));
		std::vector<Function > functions;
		functions.push_back(sphere);
		functions.push_back(elli);
		evaluate<fCMA>(functions, functions, epochs);
		evaluate<pcCMSA>(functions, functions, epochs);
		evaluate<CSA>(functions, functions, epochs);
	}
	
	
	{
		std::size_t dims=1000;
		std::size_t epochs=1000;
		Function sphere = Function(new Sphere(dims));
		std::vector<Function > functions;
		std::vector<Function > functionsEval;
		functions.push_back(Function(new NoisySphere(dims))); functionsEval.push_back(sphere);
		functions.push_back(Function(new NoisyEllipsoid(dims))); functionsEval.push_back(Function(new Ellipsoid(dims, 1.e-2)));
		evaluate<fCMA>(functions, functionsEval, epochs);
		evaluate<pcfCMA>(functions, functionsEval, epochs);
		evaluate<fCMA_NNH>(functions, functionsEval, epochs);
		evaluate<pcCMSA>(functions, functionsEval, epochs);
		evaluate<CSA>(functions, functionsEval, epochs);
	}
	
	{
		std::size_t dims=8000;
		std::size_t epochs=1000;
		Function sphere = Function(new Sphere(dims));
		std::vector<Function > functions;
		std::vector<Function > functionsEval;
		functions.push_back(Function(new NoisySphere(dims))); functionsEval.push_back(sphere);
		functions.push_back(Function(new NoisyEllipsoid(dims))); functionsEval.push_back(Function(new Ellipsoid(dims, 1.e-2)));
		evaluate<pcfCMA>(functions, functionsEval, epochs);
		evaluate<fCMA>(functions, functionsEval, epochs);
		evaluate<pcCMSA>(functions, functionsEval, epochs);
		evaluate<fCMA_NNH>(functions, functionsEval, epochs);
		evaluate<CSA>(functions, functionsEval, epochs);
	}
	
	
}
