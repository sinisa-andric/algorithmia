package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	geneticAlgorithmPopulation   = 20
	geneticAlgorithmMutationRate = 0.1
	geneticAlgorithmMaxSteps     = 1000
	geneticAlgorithmTolerance    = 1e-6
	geneticAlgorithmSpread       = 10.0
	geneticAlgorithmMutationProb = 0.3
)

// GeneticAlgorithm minimizuje konfigurisanu benchmark funkciju koristeći genetski algoritam: turnirska selekcija,
// aritmetičko ukrštanje i Gausova mutacija proizvode svaku novu generaciju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GeneticAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := geneticAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	mutationRate := geneticAlgorithmMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	maxSteps := geneticAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := geneticAlgorithmTolerance
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
			individual[d] = problem.Point[d] + (rand.Float64()*2-1)*geneticAlgorithmSpread
		}
		individuals[i] = individual
	}

	best := append([]float64(nil), problem.Point...)
	bestValue := fn.Evaluate(best)

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		nextGeneration := make([][]float64, population)

		for i := 0; i < population; i++ {
			parentA := tournamentSelect(fn, individuals)
			parentB := tournamentSelect(fn, individuals)

			child := make([]float64, dimensions)
			for d := range child {
				child[d] = (parentA[d] + parentB[d]) / 2

				if rand.Float64() < geneticAlgorithmMutationProb {
					child[d] += rand.NormFloat64() * mutationRate
				}
			}

			nextGeneration[i] = child

			value := fn.Evaluate(child)
			if value < bestValue {
				bestValue = value
				best = append([]float64(nil), child...)
			}
		}

		individuals = nextGeneration

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "genetic_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func tournamentSelect(fn functions.BenchmarkFunction, individuals [][]float64) []float64 {

	a := individuals[rand.IntN(len(individuals))]
	b := individuals[rand.IntN(len(individuals))]

	if fn.Evaluate(a) < fn.Evaluate(b) {
		return a
	}

	return b
}
