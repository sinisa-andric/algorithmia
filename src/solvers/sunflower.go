package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sunflowerPopulation = 20
	sunflowerMaxSteps   = 500
	sunflowerTolerance  = 1e-6
	sunflowerRangeLow   = -5.0
	sunflowerRangeHigh  = 5.0
)

// Sunflower minimizuje konfigurisanu benchmark funkciju koristeći Sunflower Optimization: svaki suncokret prati
// najbolje rešenje kao sunce duž normalizovanog pravca sa korakom koji opada tokom izvršavanja ali nikad ne
// dostiže nulu, uz stalno nasumično odstupanje od tog pravca
// problem.Point inicijalizuje populaciju suncokreta i određuje njenu dimenzionalnost
func Sunflower(problem models.Problem) (result models.Result, err error) {

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

	population := sunflowerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sunflowerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sunflowerTolerance
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
			individual[d] = sunflowerRangeLow + rand.Float64()*(sunflowerRangeHigh-sunflowerRangeLow)
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

		stepSize := 0.5*(1-float64(steps)/float64(maxSteps)) + 0.05

		for i := range individuals {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = best[d] - individuals[i][d]
			}
			dist := norm(diff)

			direction := make([]float64, dimensions)
			if dist > 1e-10 {
				for d := range direction {
					direction[d] = diff[d] / dist
				}
			}

			for d := range individuals[i] {
				individuals[i][d] = clamp(individuals[i][d]+direction[d]*stepSize*rand.Float64()+(rand.Float64()*2-1)*0.15, sunflowerRangeLow, sunflowerRangeHigh)
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
		Method:     "sunflower",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
