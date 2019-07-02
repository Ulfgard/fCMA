#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ShiftedObjectiveFunction_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ShiftedObjectiveFunction_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/LinAlg/rotations.h>

namespace shark {namespace benchmarks{
///  \brief Rotates an objective function using a randomly initialized rotation.
///
/// Most benchmark functions are axis aligned because it is assumed that the algorithm
/// is rotation invariant. However this does not mean that all its aspects are the same.
/// Especially linear algebra routines might take longer when the problem is not
/// axis aligned. This function creates a random rotation function and 
/// applies it to the given input points to make it no longer axis aligned.
///  \ingroup benchmarks
struct ShiftedObjectiveFunction : public SingleObjectiveFunction {
	ShiftedObjectiveFunction(SingleObjectiveFunction* objective)
	:m_objective(objective){
		if(m_objective->canProposeStartingPoint())
			m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Shifted"+m_objective->name(); }
	
	std::size_t numberOfVariables()const{
		return m_objective->numberOfVariables();
	}
	
	void init(){
		m_shift = blas::repeat(1.0, m_objective->numberOfVariables());
		m_objective->init();
	}
	
	bool hasScalableDimensionality()const{
		return m_objective->hasScalableDimensionality();
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_objective->setNumberOfVariables(numberOfVariables);
	}

	SearchPointType proposeStartingPoint() const {
		RealVector y = m_objective->proposeStartingPoint();
		
		return y+m_shift;
	}

	double eval( SearchPointType const& p ) const {
		m_evaluationCounter++;
		return m_objective->eval(p - m_shift);
	}
private:
	SingleObjectiveFunction* m_objective;
	RealVector m_shift;
};

}}

#endif
