package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	octopusPopulation = 20
	octopusMaxSteps   = 1000
	octopusTolerance  = 1e-6
	octopusRangeLow   = -5.0
	octopusRangeHigh  = 5.0
	octopusArms       = 4
)

// Octopus minimizuje konfigurisanu benchmark funkciju koristeći Octopus Swarm Optimization: svaka hobotnica
// nezavisno probnim krakovima istražuje oko sopstvenog tela, telo se pomera ka najboljem probnom kraku ako je
// bolji od trenutne pozicije, inače se telo pomera ka globalno najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Octopus(problem models.Problem) (result models.Result, err error) {

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

	population := octopusPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := octopusMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := octopusTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	octopuses := make([][]float64, population)
	values := make([]float64, population)
	for i := range octopuses {
		octopus := make([]float64, dimensions)
		for d := range octopus {
			octopus[d] = octopusRangeLow + rand.Float64()*(octopusRangeHigh-octopusRangeLow)
		}
		octopuses[i] = octopus
		values[i] = fn.Evaluate(octopus)
	}

	best := append([]float64(nil), octopuses[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), octopuses[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range octopuses {
			bestArm := octopuses[i]
			bestArmValue := values[i]

			for range octopusArms {
				tip := make([]float64, dimensions)
				for d := range tip {
					tip[d] = clamp(octopuses[i][d]+(rand.Float64()*2-1)*0.5, octopusRangeLow, octopusRangeHigh)
				}
				tipValue := fn.Evaluate(tip)
				if tipValue < bestArmValue {
					bestArmValue = tipValue
					bestArm = tip
				}
			}

			if bestArmValue < values[i] {
				for d := range octopuses[i] {
					octopuses[i][d] = octopuses[i][d] + rand.Float64()*(bestArm[d]-octopuses[i][d])
				}
			} else {
				for d := range octopuses[i] {
					octopuses[i][d] = octopuses[i][d] + rand.Float64()*(best[d]-octopuses[i][d])*0.3
				}
			}

			for d := range octopuses[i] {
				octopuses[i][d] = clamp(octopuses[i][d], octopusRangeLow, octopusRangeHigh)
			}
			values[i] = fn.Evaluate(octopuses[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), octopuses[i]...)
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
		Method:     "octopus",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
