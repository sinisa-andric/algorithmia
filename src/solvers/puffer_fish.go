package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	pufferFishPopulation = 20
	pufferFishMaxSteps   = 1000
	pufferFishTolerance  = 1e-6
	pufferFishRangeLow   = -5.0
	pufferFishRangeHigh  = 5.0
)

// PufferFish minimizuje konfigurisanu benchmark funkciju koristeći Puffer Fish Optimization: napuhuša najčešće
// normalno pliva ka najboljem rešenju, a povremeno se naduva i izvede nagli veliki nasumičan skok bežeći od pretnje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func PufferFish(problem models.Problem) (result models.Result, err error) {

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

	population := pufferFishPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := pufferFishMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := pufferFishTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	puffers := make([][]float64, population)
	values := make([]float64, population)
	for i := range puffers {
		puffer := make([]float64, dimensions)
		for d := range puffer {
			puffer[d] = pufferFishRangeLow + rand.Float64()*(pufferFishRangeHigh-pufferFishRangeLow)
		}
		puffers[i] = puffer
		values[i] = fn.Evaluate(puffer)
	}

	best := append([]float64(nil), puffers[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), puffers[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range puffers {
			if rand.Float64() < 0.85 {
				for d := range puffers[i] {
					puffers[i][d] = puffers[i][d] + rand.Float64()*(best[d]-puffers[i][d])
				}
			} else {
				for d := range puffers[i] {
					puffers[i][d] = puffers[i][d] + (rand.Float64()*2-1)*1.8
				}
			}

			for d := range puffers[i] {
				puffers[i][d] = clamp(puffers[i][d], pufferFishRangeLow, pufferFishRangeHigh)
			}
			values[i] = fn.Evaluate(puffers[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), puffers[i]...)
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
		Method:     "puffer_fish",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
