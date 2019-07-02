#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_pcCMSA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_pcCMSA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Algorithms/DirectSearch/LMCMA.h>
#include <boost/math/distributions/normal.hpp>
namespace shark {

class pcCMSA : public AbstractSingleObjectiveOptimizer<RealVector >{
public:

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "pcCMSA"; }
	
	/// \brief Calculates lambda for the supplied dimensionality n.
	static std::size_t suggestLambda( std::size_t dimension ) {
		std::size_t lambda = std::size_t( 4. + ::floor( 3 *::log( static_cast<double>( dimension ) ) ) );
		lambda += 2 - lambda % 3;//ensure lambda divisable by 3
		return lambda;
	}

	void read( InArchive & archive ){}
	void write( OutArchive & archive ) const{}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	/// \brief Initializes the algorithm for the supplied objective function.
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		SIZE_CHECK(p.size() == function.numberOfVariables());
		checkFeatures(function);
		std::vector<RealVector> points(1,p);
		std::vector<double> functionValues(1,function.eval(p));

		std::size_t lambda = pcCMSA::suggestLambda( p.size() );
		doInit(
			points,
			functionValues,
			lambda,
			3.0/std::sqrt(double(p.size()))
		);
	}

	/// \brief Initializes the algorithm for the supplied objective function.
	void init( 
		ObjectiveFunctionType const& function, 
		SearchPointType const& initialSearchPoint,
		std::size_t lambda,
		double initialSigma
	){
		std::vector<RealVector> points(1,initialSearchPoint);
		std::vector<double> functionValues(1,function.eval(initialSearchPoint));
		doInit(
			points,
			functionValues,
			lambda,
			initialSigma
		);
	}

	/// \brief Executes one iteration of the algorithm.
	void step(ObjectiveFunctionType const& function){
		std::vector<IndividualType> offspring = generateOffspring();
		//evaluate
		auto evaluator = [&](std::size_t i){
			offspring[i].unpenalizedFitness() =function(offspring[i].searchPoint());
			offspring[i].penalizedFitness() = offspring[i].unpenalizedFitness();
		};
		threading::parallelND(offspring.size(), 0, evaluator,threading::globalThreadPool());
		
		//compute the cutoff for selection
		std::vector<double> fs(m_lambda);
		for (std::size_t i = 0; i < m_lambda; i++){
			fs[i] = offspring[i].penalizedFitness();
		}
		std::sort(fs.begin(), fs.end());
		double cutoff = fs[m_mu];
		
		//update statistics
		m_mean.clear();
		m_sigma = 0.0;
		for (std::size_t i = 0; i < m_lambda; i++){
			if(offspring[i].penalizedFitness() < cutoff){
				noalias(m_mean) += offspring[i].searchPoint();
				m_sigma += offspring[i].chromosome();
			}
		}
		m_sigma /= m_mu;
		m_mean /= m_mu;
		
		//store estimate for current loss
		m_best.point = m_mean;
		m_best.value = 0.0;
		for (std::size_t i = 0; i < m_lambda; i++)
			m_best.value += offspring[i].unpenalizedFitness() / m_lambda;

		//noise handling
		if(true || function.isNoisy()){//always activated
			//store the mean function value of the last L steps
			m_Fs.push_back(function(m_mean));
			//check whether we have enough new data
			if(m_Fs.size() < m_L)
				return;
			
			//enough data, we have to check whether we made enough progress
			//if we did not make enough progress, we 
			if(detection(m_Fs, m_alphabar)){
				m_mu = std::max(m_mumin, std::size_t(m_mu / m_mudec));
			}else{
				m_mu = m_muinc * m_mu;
			}
			m_lambda = 3 * m_mu;
			//We used the data up, now we have to collect new
			m_Fs.clear();
		}
		
		
	}

	double sigma() const {
		return m_sigma;
	}
	
	std::size_t lambda() const{
		return m_lambda;
	}

protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, double> IndividualType;
	
	/// \brief Samples lambda individuals from the search distribution	
	std::vector<IndividualType> generateOffspring( ) const{
		std::vector<IndividualType> offspring(m_lambda);
		auto sampler = [&](std::size_t i){
			double& s = offspring[i].chromosome();
			RealVector& x = offspring[i].searchPoint();
			s = m_sigma * std::exp(m_tauSigma * random::gauss(random::globalRng(), 0.0, 1.0));
			auto z = remora::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, remora::cpu_tag());
			x = m_mean + s * z;
		};
		
		threading::parallelND(offspring.size(), 0, sampler,threading::globalThreadPool());
		
		return offspring;
	}

	void doInit(
		std::vector<SearchPointType> const& points,
		std::vector<ResultType> const& functionValues,
		std::size_t lambda,
		double initialSigma
	){
		SIZE_CHECK(points.size() > 0);
	
		m_numberOfVariables =points[0].size();
		m_mu = lambda / 3;
		m_mumin = m_mu;
		m_lambda = 3 * m_mu;
		
		//parameters of the sampling distribution
		m_mean = blas::repeat(0.0, m_numberOfVariables);
		m_sigma = initialSigma;
		m_tauSigma = 1.0 / std::sqrt(2.0*m_numberOfVariables);
		
		//adaptation of population size
		m_muinc = 2.0;
		m_mudec = 1.5;
		m_L = 3*m_numberOfVariables;
		m_alphabar = 0.05;//quantile for statistical test whether the progress is larger than the expected
		
		//pick starting point as best point in the set
		std::size_t pos = std::min_element(functionValues.begin(),functionValues.end())-functionValues.begin();
		m_mean = points[pos];
		m_best.point = points[pos];
		m_best.value = functionValues[pos];
	}
	
	bool detection(std::deque<double> const& Fs, double alpha) const{
		auto sign = [](double x){ return x==0? 0.5: (x < 0? -1.0: 1.0);};  
		//Mann-Kendall non-parametric test for linear trends
		std::size_t n = Fs.size();
		double S =0;
		for(std::size_t i = 0; i != n; ++i){
			for(std::size_t j = i+1; j != n; ++j){
				S += sign(Fs[j] - Fs[i]);
			}
		}
		double stdS = std::sqrt(n * (n-1) * (2*n+5) / 18.0);
		double Z = 0.0;
		if( S > 0)
			Z=(S-1)/stdS;
		else if( S < 0){
			Z=(S+1) / stdS;
		}
		boost::math::normal_distribution<> dist(0.0, 1.0 );
		return (Z <= quantile(dist, alpha));
	}
private:
	std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	std::size_t m_lambda; ///< The size of the offspring population,2*mu
	std::size_t m_mu; ///< The size of the parent population
	
	//mean of search distribution
	RealVector m_mean;
	double m_sigma;//global step-size
	double m_tauSigma;//step-size variance-kind-off

	//variables required for adaptation of population size
	std::size_t m_mumin; ///< The minimum size of the parent population
	std::size_t m_L; ///< length of history
	std::deque<double> m_Fs;///< function values of up to last L steps
	double m_muinc; ///< factor to increase population size if not enough progress was made
	double m_mudec;///< factor to decrease population size if enough progress was made
	double m_alphabar;///< quantile for hypothesis test
};
}
#endif
