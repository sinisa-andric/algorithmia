package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	shrimpPopulation = 20
	shrimpMaxSteps   = 1000
	shrimpTolerance  = 1e-6
	shrimpRangeLow   = -5.0
	shrimpRangeHigh  = 5.0
)

// Shrimp minimizuje konfigurisanu benchmark funkciju koristeći Shrimp Swarm Optimization: škampa se kreće ka
// centru mase jata i ka najboljem rešenju, uz povremeno presvlačenje — malu nasumičnu perturbaciju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Shrimp(problem models.Problem) (result models.Result, err error) {

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

	population := shrimpPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := shrimpMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := shrimpTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	shrimps := make([][]float64, population)
	values := make([]float64, population)
	for i := range shrimps {
		shrimp := make([]float64, dimensions)
		for d := range shrimp {
			shrimp[d] = shrimpRangeLow + rand.Float64()*(shrimpRangeHigh-shrimpRangeLow)
		}
		shrimps[i] = shrimp
		values[i] = fn.Evaluate(shrimp)
	}

	best := append([]float64(nil), shrimps[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), shrimps[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		center := make([]float64, dimensions)
		for _, s := range shrimps {
			for d := range center {
				center[d] += s[d]
			}
		}
		for d := range center {
			center[d] /= float64(population)
		}

		for i := range shrimps {
			for d := range shrimps[i] {
				shrimps[i][d] = shrimps[i][d] + rand.Float64()*(center[d]-shrimps[i][d]) + rand.Float64()*(best[d]-shrimps[i][d])
				if rand.Float64() < 0.1 {
					shrimps[i][d] = shrimps[i][d] + (rand.Float64()*2-1)*0.3
				}
				shrimps[i][d] = clamp(shrimps[i][d], shrimpRangeLow, shrimpRangeHigh)
			}
			values[i] = fn.Evaluate(shrimps[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), shrimps[i]...)
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
		Method:     "shrimp",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
