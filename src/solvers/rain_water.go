package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	rainWaterPopulation = 20
	rainWaterMaxSteps   = 500
	rainWaterTolerance  = 1e-6
	rainWaterRangeLow   = -5.0
	rainWaterRangeHigh  = 5.0
	rainWaterEpsilon    = 0.01
)

// RainWater minimizuje konfigurisanu benchmark funkciju koristeći Rain Water Algorithm: svaka kap teče niz
// numerički aproksimiran gradijent terena (fitnesa) brzinom koja opada ali nikad ne dostiže nulu, uz šum koji
// simulira sitne nepravilnosti terena
// problem.Point inicijalizuje populaciju kapi i određuje njenu dimenzionalnost
func RainWater(problem models.Problem) (result models.Result, err error) {

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

	population := rainWaterPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := rainWaterMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := rainWaterTolerance
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
			individual[d] = rainWaterRangeLow + rand.Float64()*(rainWaterRangeHigh-rainWaterRangeLow)
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

		flowRate := 0.3*(1-float64(steps)/float64(maxSteps)) + 0.05

		for i := range individuals {
			base := values[i]
			for d := range individuals[i] {
				perturbed := append([]float64(nil), individuals[i]...)
				perturbed[d] += rainWaterEpsilon
				gradient := (fn.Evaluate(perturbed) - base) / rainWaterEpsilon
				individuals[i][d] = clamp(individuals[i][d]-flowRate*gradient+(rand.Float64()*2-1)*0.1, rainWaterRangeLow, rainWaterRangeHigh)
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
		Method:     "rain_water",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
