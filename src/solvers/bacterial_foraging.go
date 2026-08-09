package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	bacterialForagingPopulation = 20
	bacterialForagingMaxSteps   = 1000
	bacterialForagingTolerance  = 1e-6
	bacterialForagingRangeLow   = -5.0
	bacterialForagingRangeHigh  = 5.0
	bacterialForagingStepSize   = 0.1
	bacterialForagingMaxSwim    = 4
)

// BacterialForaging minimizuje konfigurisanu benchmark funkciju koristeći Bacterial Foraging Optimization: svaka
// bakterija se pomera (tumble) u nasumičnom pravcu, i dok pomeraj u istom pravcu poboljšava rezultat nastavlja da
// pliva (swim) u tom pravcu do maksimalnog broja koraka, inače naredni put bira novi nasumičan pravac
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func BacterialForaging(problem models.Problem) (result models.Result, err error) {

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

	population := bacterialForagingPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := bacterialForagingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := bacterialForagingTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	stepSize := bacterialForagingStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	randomDirection := func() []float64 {
		dir := make([]float64, dimensions)
		sum := 0.0
		for d := range dir {
			dir[d] = rand.Float64()*2 - 1
			sum += dir[d] * dir[d]
		}
		norm := math.Sqrt(sum)
		if norm > 1e-10 {
			for d := range dir {
				dir[d] /= norm
			}
		}
		return dir
	}

	bacteria := make([][]float64, population)
	values := make([]float64, population)
	for i := range bacteria {
		bacterium := make([]float64, dimensions)
		for d := range bacterium {
			bacterium[d] = bacterialForagingRangeLow + rand.Float64()*(bacterialForagingRangeHigh-bacterialForagingRangeLow)
		}
		bacteria[i] = bacterium
		values[i] = fn.Evaluate(bacterium)
	}

	best := append([]float64(nil), bacteria[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), bacteria[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range bacteria {
			direction := randomDirection()

			candidate := make([]float64, dimensions)
			for d := range candidate {
				candidate[d] = clamp(bacteria[i][d]+stepSize*direction[d], bacterialForagingRangeLow, bacterialForagingRangeHigh)
			}
			candidateValue := fn.Evaluate(candidate)

			if candidateValue < values[i] {
				bacteria[i] = candidate
				values[i] = candidateValue

				for range bacterialForagingMaxSwim {
					next := make([]float64, dimensions)
					for d := range next {
						next[d] = clamp(bacteria[i][d]+stepSize*direction[d], bacterialForagingRangeLow, bacterialForagingRangeHigh)
					}
					nextValue := fn.Evaluate(next)
					if nextValue < values[i] {
						bacteria[i] = next
						values[i] = nextValue
					} else {
						break
					}
				}
			}
			// inače ostaje na trenutnoj poziciji — naredni tumble bira nov nasumičan pravac
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), bacteria[i]...)
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
		Method:     "bacterial_foraging",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
