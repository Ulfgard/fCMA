#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_pcfCMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_pcfCMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Algorithms/DirectSearch/LMCMA.h>
#include <boost/math/distributions/normal.hpp>
namespace shark {

class pcfCMA : public AbstractSingleObjectiveOptimizer<RealVector >{
public:

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "pcfCMA"; }
	
	/// \brief Calculates lambda for the supplied dimensionality n.
	static std::size_t suggestLambda( std::size_t dimension ) {
		return std::size_t( 4. + ::floor( 3 *::log( static_cast<double>( dimension ) ) ) );
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

		std::size_t lambda = pcfCMA::suggestLambda( p.size() );
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
		//evaluate points and average over the number of re-evaluations
		
		auto evaluator = [&](std::size_t i){
			offspring[i].unpenalizedFitness() = 0.0;
			for(std::size_t t = 0; t != m_numEvals; ++t)
				offspring[i].unpenalizedFitness() += function(offspring[i].searchPoint()) / m_numEvals;
			offspring[i].penalizedFitness() = offspring[i].unpenalizedFitness();
		};
		
		threading::parallelND(offspring.size(), 0, evaluator,threading::globalThreadPool());
		
		updatePopulation(offspring);
		
		//noise handling
		if(function.isNoisy()){
			//store the mean function value of the last L steps
			m_Fs.push_back(function(m_mean));
			//check whether we have enough new data
			if(m_Fs.size() < m_L)
				return;
			
			//enough data, we have to check whether we made enough progress
			//if we did not make enough progress, we 
			if(detection(m_Fs, m_alphabar)){
				m_numEvals = std::max<std::size_t>(1, std::size_t(m_numEvals / m_evalsDec));
			}else{
				m_numEvals = m_evalsInc * m_numEvals;
			}
			//We used the data up, now we have to collect new
			m_Fs.clear();
		}
	}

	double sigma() const {
		return m_sigma;
	}
	std::size_t lambda() const{
		return m_numEvals;
	}

protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, RealVector> IndividualType;

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
	
	/// \brief Samples lambda individuals from the search distribution	
	std::vector<IndividualType> generateOffspring( ) const{
		std::vector<IndividualType> offspring(m_lambda);
		auto sampler = [&](std::size_t i){
			RealVector& z = offspring[i].chromosome();
			RealVector& x = offspring[i].searchPoint();
			z = remora::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, remora::cpu_tag());
			x = m_mean + m_sigma * z;
		};
		
		threading::parallelND(offspring.size(), 0, sampler,threading::globalThreadPool());
		
		return offspring;
	}

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	void updatePopulation( std::vector<IndividualType > const& offspring){	
		//compute the weights
		RealVector weights(m_lambda, 0.0);
		for (std::size_t i = 0; i < m_lambda; i++){
			weights(i) = -offspring[i].penalizedFitness();
		}
		weights -= min(weights);
		weights /= norm_1(weights);
		
		//update learning rates
		double cPath = 2.0 * (m_muEff + 2.)/(m_numberOfVariables + m_muEff + 5.);
		double dPath = 2.0 * cPath/std::sqrt(cPath * (2-cPath));
		double cmuEff = 0.01;
		//first iteration: initialize all paths with true data from the function
		if(m_firstIter){
			m_firstIter = false;
			cmuEff = 1.0;
		}

		//gradient of mean
		RealVector dMean( m_numberOfVariables, 0. );
		RealVector stepZ( m_numberOfVariables, 0. );
		for (std::size_t i = 0; i < m_lambda; i++){
			noalias(dMean) += (weights(i) - 1.0/m_lambda) * offspring[i].searchPoint();
			noalias(stepZ) += weights(i) * offspring[i].chromosome();
		}

		noalias(m_path)= (1-cPath) * m_path + std::sqrt(cPath * (2-cPath) * m_muEff) * stepZ;
		m_gammaPath = sqr(1-cPath) * m_gammaPath+ cPath * (2-cPath);
		double deviationStepLen = norm_2(m_path)/std::sqrt(m_numberOfVariables) - std::sqrt(m_gammaPath);
		
		//performing steps in variables
		noalias(m_mean) +=  dMean;
		
		m_sigma *= std::exp(deviationStepLen*dPath);
		m_muEff = (1-cmuEff)* m_muEff + cmuEff / sum(sqr(weights));
		//store estimate for current loss
		m_best.point = m_mean;
		m_best.value = 0.0;
		for (std::size_t i = 0; i < m_lambda; i++)
			m_best.value += offspring[i].unpenalizedFitness()/m_lambda;
	}

	void doInit(
		std::vector<SearchPointType> const& points,
		std::vector<ResultType> const& functionValues,
		std::size_t lambda,
		double initialSigma
	){
		SIZE_CHECK(points.size() > 0);
	
		m_numberOfVariables =points[0].size();
		m_lambda = lambda;
		m_numEvals = 1;
		m_firstIter = true;
		
		//variables for mean
		m_mean = blas::repeat(0.0, m_numberOfVariables);
		
		//variables for step size
		m_path = blas::repeat(0.0, m_numberOfVariables);
		m_sigma = sqr(initialSigma);
		m_muEff = 0.0;
		m_gammaPath = 0.0;
		
		//adaptation of population size
		m_evalsInc = 2.0;
		m_evalsDec = 1.5;
		m_L = 100;
		m_alphabar = 0.05;//quantile for statistical test whether the progress is larger than the expected
		m_Fs.clear();
		
		//pick starting point as best point in the set
		std::size_t pos = std::min_element(functionValues.begin(),functionValues.end())-functionValues.begin();
		m_mean = points[pos];
		m_best.point = points[pos];
		m_best.value = functionValues[pos];
	}
private:
	std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	std::size_t m_lambda; ///< The size of the offspring population, needs to be larger than mu.
	

	//mean of search distribution
	RealVector m_mean;
	
	//Variables governing step size update
	RealVector m_path;
	double m_gammaPath;

	double m_sigma;//global step-size
	double m_muEff;
	bool m_firstIter;

	//variables required for adaptation of population size
	std::size_t m_numEvals; ///< The number of re-evaluations per point
	std::size_t m_L; ///< length of history
	std::deque<double> m_Fs;///< function values of up to last L steps
	double m_evalsInc; ///< factor to increase population size if not enough progress was made
	double m_evalsDec;///< factor to decrease population size if enough progress was made
	double m_alphabar;///< quantile for hypothesis test
};
}
#endif
