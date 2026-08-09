package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	peregrineFalconPopulation = 20
	peregrineFalconMaxSteps   = 1000
	peregrineFalconTolerance  = 1e-6
	peregrineFalconRangeLow   = -5.0
	peregrineFalconRangeHigh  = 5.0
)

// PeregrineFalcon minimizuje konfigurisanu benchmark funkciju koristeći Peregrine Falcon Optimization: sokol se ili
// obrušava ka najboljem rešenju brzinom koja opada tokom izvršavanja, ili kruži istražujući prostor pretrage
// nasumičnim uglom
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func PeregrineFalcon(problem models.Problem) (result models.Result, err error) {

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

	population := peregrineFalconPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := peregrineFalconMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := peregrineFalconTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	falcons := make([][]float64, population)
	values := make([]float64, population)
	for i := range falcons {
		falcon := make([]float64, dimensions)
		for d := range falcon {
			falcon[d] = peregrineFalconRangeLow + rand.Float64()*(peregrineFalconRangeHigh-peregrineFalconRangeLow)
		}
		falcons[i] = falcon
		values[i] = fn.Evaluate(falcon)
	}

	best := append([]float64(nil), falcons[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), falcons[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		stoopFactor := 2 * (1 - float64(steps)/float64(maxSteps))

		for i := range falcons {
			if rand.Float64() < 0.6 {
				for d := range falcons[i] {
					falcons[i][d] = falcons[i][d] + stoopFactor*rand.Float64()*(best[d]-falcons[i][d])
				}
			} else {
				angle := rand.Float64() * 2 * math.Pi
				for d := range falcons[i] {
					falcons[i][d] = falcons[i][d] + 0.5*math.Cos(angle)*(rand.Float64()*2-1)
				}
			}

			for d := range falcons[i] {
				falcons[i][d] = clamp(falcons[i][d], peregrineFalconRangeLow, peregrineFalconRangeHigh)
			}
			values[i] = fn.Evaluate(falcons[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), falcons[i]...)
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
		Method:     "peregrine_falcon",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
