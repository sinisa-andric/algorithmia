package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	gravitationalLocalSearchPopulation = 20
	gravitationalLocalSearchMaxSteps   = 500
	gravitationalLocalSearchTolerance  = 1e-6
	gravitationalLocalSearchRangeLow   = -5.0
	gravitationalLocalSearchRangeHigh  = 5.0
)

// GravitationalLocalSearch minimizuje konfigurisanu benchmark funkciju koristeći Gravitational Local Search:
// svaka jedinka je gravitaciono privučena isključivo ka best-u (bez masa ili N-tela interakcije sa ostalim
// jedinkama kao kod gravitational_search.go), sa konstantom koja opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GravitationalLocalSearch(problem models.Problem) (result models.Result, err error) {

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

	population := gravitationalLocalSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := gravitationalLocalSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gravitationalLocalSearchTolerance
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
			individual[d] = gravitationalLocalSearchRangeLow + rand.Float64()*(gravitationalLocalSearchRangeHigh-gravitationalLocalSearchRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

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

		g := 1.0*(1-float64(steps)/float64(maxSteps)) + 0.05

		for i := range individuals {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = best[d] - individuals[i][d]
			}
			r := norm(diff) + 1e-10

			for d := range individuals[i] {
				accel := g * diff[d] / r
				individuals[i][d] = clamp(individuals[i][d]+accel*rand.Float64()+(rand.Float64()*2-1)*0.1, gravitationalLocalSearchRangeLow, gravitationalLocalSearchRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		Method:     "gravitational_local_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
