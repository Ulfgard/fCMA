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
#include "CSA.h"
#include "pcCMSA.h"
#include "pcfCMA.h"
//benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

//misc
#include <fstream>
#include <boost/filesystem.hpp>
using namespace shark;

struct LogSphere : public SingleObjectiveFunction {
	
	LogSphere(std::size_t numberOfVariables):m_numberOfVariables(numberOfVariables){
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LogSphere"; }

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
		
		return std::log(norm_sqr(x)+1.e-10);
	}
private:
	std::size_t m_numberOfVariables;
};

struct LogDiscus: public SingleObjectiveFunction {
	
	LogDiscus(std::size_t numberOfVariables,double alpha = 1.E-3):m_numberOfVariables(numberOfVariables), m_alpha(alpha){
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LogDiscus"; }

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
		
		double sum =  sqr(x(0));
		for (std::size_t i = 1; i < x.size(); i++)
			sum += m_alpha * sqr(x(i));
		
		return std::log(sum+1.e-10);
	}
private:
	std::size_t m_numberOfVariables;
	double m_alpha;
};
		


typedef boost::shared_ptr<SingleObjectiveFunction > Function;
template<class Optimizer>
void evaluate(std::vector<std::pair<Function, double> > const& functions, std::size_t trials){
	Optimizer cma;
	random::globalRng().seed(142);
	boost::filesystem::create_directory("./results/"+cma.name());
	std::string fileName="./results/"+cma.name()+"/runtime.txt";
	std::ofstream str(fileName.c_str());
	for(std::size_t d = 1000; d< 16001; d *= 2){
		str<<d;
		threading::mapApply(
			functions,
			[&](std::pair<Function, double> const& pair){
				auto const& f = pair.first;
				double fstop = pair.second;
				f->setNumberOfVariables(d);
				f->init();
			
				for(std::size_t i = 0; i != trials; ++i){
					//initialize the optimizer
					Optimizer cma;
					cma.init( *f);
					
					while(cma.solution().value > fstop)
						cma.step(*f);
				}
				return f->evaluationCounter()/trials;
			},
			[&](std::size_t iters){
				str<<"\t"<<iters;
				std::cout<<d<<"\t"<<iters/d<<std::endl;
			},
			threading::globalThreadPool()
		);
		str<<std::endl;
	}
}

int main( int argc, char ** argv ) {
	boost::filesystem::create_directory("./results");
	using namespace shark::benchmarks;
	std::vector<std::pair<Function, double> > functions;
	functions.reserve(100);
	functions.emplace_back(Function(new Sphere(5)), 1.e-7);
	functions.emplace_back(Function(new Discus(5,1.e-2)), 1.e-7);
	functions.emplace_back(Function(new Cigar(5,1.e-2)), 1.e-7);
	functions.emplace_back(Function(new Ellipsoid(5,1.e-2)), 1.e-7);
	functions.emplace_back(Function(new LogSphere(5)), std::log(1.e-7));
	functions.emplace_back(Function(new LogDiscus(5,1.e-2)), std::log(1.e-7));
	evaluate<fCMA>(functions, 10);
	evaluate<pcCMSA>(functions, 10);
	evaluate<pcfCMA>(functions, 10);
	evaluate<CSA>(functions, 10);
}
