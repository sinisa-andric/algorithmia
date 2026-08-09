package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	pigPopulation = 20
	pigMaxSteps   = 1000
	pigTolerance  = 1e-6
	pigRangeLow   = -5.0
	pigRangeHigh  = 5.0
)

// Pig minimizuje konfigurisanu benchmark funkciju koristeći Pig-inspired Optimization: svinja ili njuška oko svoje
// trenutne pozicije (lokalna pretraga amplitudom koja opada tokom izvršavanja) ili se premešta ka najboljem
// pronađenom izvoru hrane
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Pig(problem models.Problem) (result models.Result, err error) {

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

	population := pigPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := pigMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := pigTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	pigs := make([][]float64, population)
	values := make([]float64, population)
	for i := range pigs {
		pig := make([]float64, dimensions)
		for d := range pig {
			pig[d] = pigRangeLow + rand.Float64()*(pigRangeHigh-pigRangeLow)
		}
		pigs[i] = pig
		values[i] = fn.Evaluate(pig)
	}

	best := append([]float64(nil), pigs[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), pigs[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range pigs {
			if rand.Float64() < 0.6 {
				for d := range pigs[i] {
					pigs[i][d] = pigs[i][d] + (rand.Float64()*2-1)*(1-float64(steps)/float64(maxSteps))
				}
			} else {
				for d := range pigs[i] {
					pigs[i][d] = pigs[i][d] + rand.Float64()*(best[d]-pigs[i][d])
				}
			}

			for d := range pigs[i] {
				pigs[i][d] = clamp(pigs[i][d], pigRangeLow, pigRangeHigh)
			}
			values[i] = fn.Evaluate(pigs[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), pigs[i]...)
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
		Method:     "pig",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
