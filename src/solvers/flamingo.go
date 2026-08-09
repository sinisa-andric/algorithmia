package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	flamingoPopulation = 20
	flamingoMaxSteps   = 1000
	flamingoTolerance  = 1e-6
	flamingoRangeLow   = -5.0
	flamingoRangeHigh  = 5.0
)

// Flamingo minimizuje konfigurisanu benchmark funkciju koristeći Flamingo Search Algorithm: flamingo se naizmenično
// migratorno kreće ka najboljem rešenju ili se lokalno hrani filtrirajući mulj,
// uz amplitudu ispaše koja opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Flamingo(problem models.Problem) (result models.Result, err error) {

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

	population := flamingoPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := flamingoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := flamingoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	flamingos := make([][]float64, population)
	values := make([]float64, population)
	for i := range flamingos {
		flamingo := make([]float64, dimensions)
		for d := range flamingo {
			flamingo[d] = flamingoRangeLow + rand.Float64()*(flamingoRangeHigh-flamingoRangeLow)
		}
		flamingos[i] = flamingo
		values[i] = fn.Evaluate(flamingo)
	}

	best := append([]float64(nil), flamingos[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), flamingos[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range flamingos {
			if rand.Float64() < 0.5 {
				for d := range flamingos[i] {
					flamingos[i][d] = flamingos[i][d] + rand.Float64()*(best[d]-flamingos[i][d])
				}
			} else {
				for d := range flamingos[i] {
					flamingos[i][d] = flamingos[i][d] + (rand.Float64()*2-1)*0.5*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range flamingos[i] {
				flamingos[i][d] = clamp(flamingos[i][d], flamingoRangeLow, flamingoRangeHigh)
			}
			values[i] = fn.Evaluate(flamingos[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), flamingos[i]...)
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
		Method:     "flamingo",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
