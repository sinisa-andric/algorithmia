package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	memeticAlgorithmPopulation    = 20
	memeticAlgorithmMutationRate  = 0.1
	memeticAlgorithmLocalSteps    = 10
	memeticAlgorithmMaxSteps      = 500
	memeticAlgorithmTolerance     = 1e-6
	memeticAlgorithmSpread        = 10.0
	memeticAlgorithmMutationProb  = 0.3
	memeticAlgorithmLocalStepSize = 0.01
)

// MemeticAlgorithm minimizuje sphere funkciju koristeći memetički genetski algoritam: turnirska selekcija i
// aritmetičko ukrštanje proizvode svako dete, koje se zatim dorađuje lokalnim gradijentnim spustom
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MemeticAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := memeticAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	mutationRate := memeticAlgorithmMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	localSteps := memeticAlgorithmLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	maxSteps := memeticAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := memeticAlgorithmTolerance
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
			individual[d] = problem.Point[d] + (rand.Float64()*2-1)*memeticAlgorithmSpread
		}
		individuals[i] = localSearch(fn, individual, localSteps)
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
				child[d] = (parentA[d] + parentB[d]) / 2

				if rand.Float64() < memeticAlgorithmMutationProb {
					child[d] += rand.NormFloat64() * mutationRate
				}
			}

			child = localSearch(fn, child, localSteps)
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
		Method:     "memetic_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func localSearch(fn functions.BenchmarkFunction, point []float64, steps int) []float64 {

	point = append([]float64(nil), point...)

	for i := 0; i < steps; i++ {
		gradient := fn.Gradient(point)
		for d := range point {
			point[d] -= memeticAlgorithmLocalStepSize * gradient[d]
		}
	}

	return point
}
