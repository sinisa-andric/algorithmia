package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	forestOptimizationPopulation = 20
	forestOptimizationMaxSteps   = 500
	forestOptimizationAgeLimit   = 5
	forestOptimizationTolerance  = 1e-6
	forestOptimizationRangeLow   = -5.0
	forestOptimizationRangeHigh  = 5.0
)

// ForestOptimization minimizuje konfigurisanu benchmark funkciju koristeći Forest Optimization Algorithm: mlada
// stabla lokalno pretražuju sopstvenu okolinu i stare bez poboljšanja, a stabla koja dostignu starosnu granicu se
// seku i zasejavaju iznova u nasumičnoj okolini najboljeg rešenja
// problem.Point inicijalizuje populaciju stabala i određuje njenu dimenzionalnost
func ForestOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := forestOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := forestOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	ageLimit := forestOptimizationAgeLimit
	if v, ok := problem.Payload["age_limit"].(float64); ok {
		ageLimit = int(v)
	}

	tolerance := forestOptimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = forestOptimizationRangeLow + rand.Float64()*(forestOptimizationRangeHigh-forestOptimizationRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	// age je indeksiran po poziciji u individuals/values i čuva se između koraka za istu jedinku — ova populacija
	// se nikad ne sortira niti menja veličinu tokom izvršavanja, pa indeks i uvek referencira isto stablo
	age := make([]int, population)

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range individuals {
			if age[i] < ageLimit {
				novo := make([]float64, dimensions)
				for d := range novo {
					novo[d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.3, forestOptimizationRangeLow, forestOptimizationRangeHigh)
				}
				novoValue := fn.Evaluate(novo)
				if novoValue < values[i] {
					individuals[i] = novo
					values[i] = novoValue
					age[i] = 0
				} else {
					age[i]++
				}
			} else {
				individual := make([]float64, dimensions)
				for d := range individual {
					individual[d] = clamp(best[d]+(rand.Float64()*2-1)*0.8, forestOptimizationRangeLow, forestOptimizationRangeHigh)
				}
				individuals[i] = individual
				values[i] = fn.Evaluate(individual)
				age[i] = 0
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "forest_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
