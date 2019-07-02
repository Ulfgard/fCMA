#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CSA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CSA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Algorithms/DirectSearch/LMCMA.h>
namespace shark {

	
//CMA approximation with C=sigma*(I+sigma_pc*p_cp_c^T)
//for optimizing the PAC bound which regularizes the search distribution
//for penalty term see step()
class CSA : public AbstractSingleObjectiveOptimizer<RealVector >{
public:
	double chi( unsigned int n ) {
		return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
	}
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CSA"; }

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

		std::size_t lambda = CSA::suggestLambda( p.size() );
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
		
		
		// Selection
		std::vector< IndividualType > parents( m_mu );
		ElitistSelection<IndividualType::FitnessOrdering> selection;
		selection(offspring.begin(),offspring.end(),parents.begin(), parents.end());
		updatePopulation(parents);
	}

	double sigma() const {
		return m_sigma;
	}
	std::size_t lambda() const{
		return m_lambda;
	}

protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, RealVector> IndividualType;
	
	/// \brief Samples lambda individuals from the search distribution	
	std::vector<IndividualType> & generateOffspring( ) const{
		auto sampler = [&](std::size_t i){
			RealVector& z = m_offspring[i].chromosome();
			RealVector& x = m_offspring[i].searchPoint();
			noalias(z) = remora::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, remora::cpu_tag());
			noalias(x) = m_mean + m_sigma * z;
		};
		
		threading::parallelND(m_offspring.size(), 0, sampler,threading::globalThreadPool());
		
		return m_offspring;
	}

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	void updatePopulation( std::vector<IndividualType > const& offspring){	
		RealVector z(m_numberOfVariables,0.0);
		RealVector m(m_numberOfVariables,0.0);
		for(std::size_t i = 0; i != m_mu; ++i){
			noalias(z) += m_weights(i)*offspring[i].chromosome();
			noalias(m) += m_weights(i)*offspring[i].searchPoint();
		}
		RealVector y = (m - m_mean) / m_sigma;

		// Step size update
		m_evolutionPathSigma = (1. - m_cSigma)*m_evolutionPathSigma + std::sqrt( m_cSigma * (2. - m_cSigma) * m_muEff ) * z;
		m_gammaPath = sqr(1-m_cSigma) * m_gammaPath+ m_cSigma * (2-m_cSigma);
		m_sigma *= std::exp( (m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma)/ chi( m_numberOfVariables ) - std::sqrt(m_gammaPath)) );
		m_mean = m;
		
		//store estimate for current loss
		m_best.point = m_mean;
		m_best.value = 0.0;
		for (std::size_t i = 0; i < m_mu; i++)
			m_best.value += offspring[i].unpenalizedFitness()/m_mu;
	}

	void doInit(
		std::vector<SearchPointType> const& points,
		std::vector<ResultType> const& functionValues,
		std::size_t lambda,
		double initialSigma
	){
		SIZE_CHECK(points.size() > 0);
	
		m_numberOfVariables = points[0].size();
		double d = (double)m_numberOfVariables;
		m_lambda = lambda;
		m_mu = lambda/2;
		m_sigma = initialSigma;

		m_mean = blas::repeat(0.0,m_numberOfVariables);
		m_evolutionPathSigma = blas::repeat(0.0,m_numberOfVariables);

		//weighting of the k-best individuals
		m_weights.resize(m_mu);
		for (unsigned int i = 0; i < m_mu; i++)
			m_weights(i) = ::log(m_mu + 0.5) - ::log(1. + i); // eq. (45)
		m_weights /= sum(m_weights); // eq. (45)
		m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))

		// Step size control
		m_cSigma = 2*(m_muEff + 2.)/(d + m_muEff + 5.); // eq. (46)
		m_dSigma = 1. + m_cSigma; // eq. (46)
		m_dSigma /=4;
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
	unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
	unsigned int m_mu; ///< The size of the parent population.
	unsigned int m_lambda; ///< The size of the offspring population, needs to be larger than mu.
	
	double m_sigma; 
	double m_cSigma;
	double m_dSigma;
	double m_muEff;
	double m_gammaPath;
	
	RealVector m_mean;
	RealVector m_weights;
	RealVector m_evolutionPathSigma;
};
}
#endif
