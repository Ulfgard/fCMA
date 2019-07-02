#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_NNH_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_NNH_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Algorithms/DirectSearch/LMCMA.h>
namespace shark {

	
//CMA approximation with C=sigma*(I+sigma_pc*p_cp_c^T)
//for optimizing the PAC bound which regularizes the search distribution
//for penalty term see step()
class fCMA_NNH : public AbstractSingleObjectiveOptimizer<RealVector >{
public:

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "fCMANNH"; }
	
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

		std::size_t lambda = fCMA_NNH::suggestLambda( p.size() );
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
		std::vector<IndividualType>& offspring = generateOffspring();
		//evaluate 
		auto evaluator = [&](std::size_t i){
			offspring[i].unpenalizedFitness() =function(offspring[i].searchPoint());
			offspring[i].penalizedFitness() = offspring[i].unpenalizedFitness();
		};
		
		threading::parallelND(offspring.size(), 0, evaluator,threading::globalThreadPool());
		updatePopulation(offspring);
	}

	double sigma() const {
		return std::sqrt(m_var);
	}
	std::size_t lambda() const{
		return m_lambda;
	}

protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, RealVector> IndividualType;
	
	/// \brief Samples lambda individuals from the search distribution	
	std::vector<IndividualType> & generateOffspring( ) const{
		double sigma = std::sqrt(m_var);
		auto sampler = [&](std::size_t i){
			RealVector& z = m_offspring[i].chromosome();
			RealVector& x = m_offspring[i].searchPoint();
			noalias(z) = remora::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, remora::cpu_tag());
			noalias(x) = m_mean + sigma * z;
		};
		
		threading::parallelND(m_offspring.size(), 0, sampler,threading::globalThreadPool());
		
		return m_offspring;
	}

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	void updatePopulation( std::vector<IndividualType > const& offspring){	
		//compute the weights
		RealVector weights(m_lambda, 0.0);
		for (std::size_t i = 0; i < m_lambda; i++){
			weights(i) = -offspring[i].penalizedFitness();
		}
		weights -=min(weights);
		weights /= norm_1(weights);
		
		//update learning rates
		double cPath = 2.0*(m_muEff + 2.)/(m_numberOfVariables + m_muEff + 5.);
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
		
		//gradient of gamma
		noalias(m_path)= (1-cPath) * m_path + std::sqrt(cPath * (2-cPath) * m_muEff) * stepZ;
		m_gammaPath = sqr(1-cPath) * m_gammaPath+ cPath * (2-cPath);
		double deviationStepLen = norm_2(m_path)/std::sqrt(m_numberOfVariables) - std::sqrt(m_gammaPath);
		
		//performing steps in variables
		noalias(m_mean) += dMean;
		m_var *= std::exp(deviationStepLen*dPath);
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
		
		m_firstIter = true;
		
		//variables for mean
		m_mean = blas::repeat(0.0, m_numberOfVariables);
		
		//variables for step size
		m_path = blas::repeat(0.0, m_numberOfVariables);
		m_var = sqr(initialSigma);
		m_muEff = 0.0;
		m_gammaPath = 0.0;
		
		//pick starting point as best point in the set
		std::size_t pos = std::min_element(functionValues.begin(),functionValues.end())-functionValues.begin();
		m_mean = points[pos];
		m_best.point = points[pos];
		m_best.value = functionValues[pos];
		
		//initialize offspring array
		m_offspring.resize(m_lambda);
		for( std::size_t i = 0; i < m_offspring.size(); i++ ) {
			m_offspring[i].chromosome()  = blas::repeat(0.0, m_numberOfVariables);
			m_offspring[i].searchPoint()  = blas::repeat(0.0, m_numberOfVariables);
		}
	}
private:
	mutable std::vector<IndividualType > m_offspring;
	std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	std::size_t m_lambda; ///< The size of the offspring population, needs to be larger than mu.

	//mean of search distribution
	RealVector m_mean;
	double m_gammaPath;
	//Variables governing step size update
	RealVector m_path;
	double m_var;//global step-size
	double m_muEff;

	bool m_firstIter;
};
}
#endif
