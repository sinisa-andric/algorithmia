package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sharkSmellPopulation = 20
	sharkSmellMaxSteps   = 1000
	sharkSmellTolerance  = 1e-6
	sharkSmellRangeLow   = -5.0
	sharkSmellRangeHigh  = 5.0
	sharkSmellEpsilon    = 1e-4
	sharkSmellEta        = 0.3
)

// SharkSmell minimizuje konfigurisanu benchmark funkciju koristeći Shark Smell Optimization: svaka ajkula prati
// numerički aproksimirani gradijent mirisa niz njegov pravac (spust), a zatim dodaje malu rotacionu perturbaciju
// oko nove pozicije radi održavanja istraživanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SharkSmell(problem models.Problem) (result models.Result, err error) {

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

	population := sharkSmellPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sharkSmellMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sharkSmellTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sharks := make([][]float64, population)
	values := make([]float64, population)
	for i := range sharks {
		shark := make([]float64, dimensions)
		for d := range shark {
			shark[d] = sharkSmellRangeLow + rand.Float64()*(sharkSmellRangeHigh-sharkSmellRangeLow)
		}
		sharks[i] = shark
		values[i] = fn.Evaluate(shark)
	}

	best := append([]float64(nil), sharks[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), sharks[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range sharks {
			base := fn.Evaluate(sharks[i])
			pseudoGradient := make([]float64, dimensions)
			for d := range pseudoGradient {
				perturbed := append([]float64(nil), sharks[i]...)
				perturbed[d] += sharkSmellEpsilon
				pseudoGradient[d] = (fn.Evaluate(perturbed) - base) / sharkSmellEpsilon
			}

			for d := range sharks[i] {
				sharks[i][d] = sharks[i][d] - sharkSmellEta*pseudoGradient[d]
				sharks[i][d] = sharks[i][d] + (rand.Float64()*2-1)*0.1
				sharks[i][d] = clamp(sharks[i][d], sharkSmellRangeLow, sharkSmellRangeHigh)
			}
			values[i] = fn.Evaluate(sharks[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), sharks[i]...)
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
		Method:     "shark_smell",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
