package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	gaLocalSearchPopulation     = 20
	gaLocalSearchMutationRate   = 0.1
	gaLocalSearchLocalSteps     = 10
	gaLocalSearchMaxSteps       = 500
	gaLocalSearchTolerance      = 1e-6
	gaLocalSearchSpread         = 10.0
	gaLocalSearchLocalLearnRate = 0.05
)

// GALocalSearch minimizuje sphere funkciju koristeći memetički genetski algoritam: turnirska selekcija i aritmetičko
// ukrštanje proizvode svako dete, primenjuje se Gausova mutacija, a zatim se svako dete dorađuje sa nekoliko koraka
// lokalnog gradijentnog spusta
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GALocalSearch(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	population := gaLocalSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	mutationRate := gaLocalSearchMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	localSteps := gaLocalSearchLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	maxSteps := gaLocalSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gaLocalSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = problem.Point[d] + (rand.Float64()*2-1)*gaLocalSearchSpread
		}
		individuals[i] = individual
	}

	best := append([]float64(nil), problem.Point...)
	bestValue := fn.Evaluate(best)
	for _, individual := range individuals {
		if value := fn.Evaluate(individual); value < bestValue {
			bestValue = value
			best = append([]float64(nil), individual...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		nextGeneration := make([][]float64, population)

		for i := 0; i < population; i++ {
			parentA := tournamentSelect(fn, individuals)
			parentB := tournamentSelect(fn, individuals)

			child := make([]float64, dimensions)
			for d := range child {
				child[d] = (parentA[d]+parentB[d])/2 + randNorm()*mutationRate
			}

			for s := 0; s < localSteps; s++ {
				gradient := fn.Gradient(child)
				for d := range child {
					child[d] -= gaLocalSearchLocalLearnRate * gradient[d]
				}
			}

			nextGeneration[i] = child

			if value := fn.Evaluate(child); value < bestValue {
				bestValue = value
				best = append([]float64(nil), child...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}

		individuals = nextGeneration
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "ga_local_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
