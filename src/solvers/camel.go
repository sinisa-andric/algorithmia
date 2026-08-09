package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	camelPopulation = 20
	camelMaxSteps   = 1000
	camelTolerance  = 1e-6
	camelRangeLow   = -5.0
	camelRangeHigh  = 5.0
)

// Camel minimizuje konfigurisanu benchmark funkciju koristeći Camel Algorithm: izdržljivost svake kamile ponderiše
// kretanje ka najboljem rešenju nasuprot nasumičnoj perturbaciji, a temperatura koja raste tokom izvršavanja
// postepeno smanjuje prosečnu izdržljivost
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Camel(problem models.Problem) (result models.Result, err error) {

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

	population := camelPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := camelMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := camelTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	camels := make([][]float64, population)
	values := make([]float64, population)
	for i := range camels {
		camel := make([]float64, dimensions)
		for d := range camel {
			camel[d] = camelRangeLow + rand.Float64()*(camelRangeHigh-camelRangeLow)
		}
		camels[i] = camel
		values[i] = fn.Evaluate(camel)
	}

	best := append([]float64(nil), camels[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), camels[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		temperature := float64(steps) / float64(maxSteps)
		endurance := (1 - temperature) * rand.Float64()

		for i := range camels {
			for d := range camels[i] {
				camels[i][d] = camels[i][d] + endurance*(best[d]-camels[i][d]) + (1-endurance)*(rand.Float64()*2-1)*0.3
				camels[i][d] = clamp(camels[i][d], camelRangeLow, camelRangeHigh)
			}
			values[i] = fn.Evaluate(camels[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), camels[i]...)
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
		Method:     "camel",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
