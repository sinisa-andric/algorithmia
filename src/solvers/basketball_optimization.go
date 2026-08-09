package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	basketballOptimizationPopulation = 20
	basketballOptimizationMaxSteps   = 500
	basketballOptimizationTolerance  = 1e-6
	basketballOptimizationRangeLow   = -5.0
	basketballOptimizationRangeHigh  = 5.0
)

// BasketballOptimization minimizuje konfigurisanu benchmark funkciju koristeći Basketball Optimization Algorithm:
// igrači šutiraju ka košu (best-u) sa preciznošću koja raste vežbanjem tokom izvršavanja, a preostali deo
// kretanja ostaje nasumična promašena putanja
// problem.Point inicijalizuje populaciju igrača i određuje njenu dimenzionalnost
func BasketballOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := basketballOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := basketballOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := basketballOptimizationTolerance
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
			individual[d] = basketballOptimizationRangeLow + rand.Float64()*(basketballOptimizationRangeHigh-basketballOptimizationRangeLow)
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

		accuracy := 0.3 + 0.6*float64(steps)/float64(maxSteps)

		for i := range individuals {
			for d := range individuals[i] {
				pull := accuracy * rand.Float64() * (best[d] - individuals[i][d])
				noise := (1 - accuracy) * (rand.Float64()*2 - 1) * 0.5
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, basketballOptimizationRangeLow, basketballOptimizationRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
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
		Method:     "basketball_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
