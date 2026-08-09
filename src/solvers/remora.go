package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	remoraPopulation = 20
	remoraMaxSteps   = 1000
	remoraTolerance  = 1e-6
	remoraRangeLow   = -5.0
	remoraRangeHigh  = 5.0
)

// Remora minimizuje konfigurisanu benchmark funkciju koristeći Remora Optimization Algorithm: svaka remora se ili
// prianja za nasumičnog domaćina krećući se ka njemu i ka najboljem rešenju,
// ili slobodno pliva kombinujući privlačenje ka najboljem i ka drugoj nasumičnoj remori
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Remora(problem models.Problem) (result models.Result, err error) {

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

	population := remoraPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := remoraMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := remoraTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	remoras := make([][]float64, population)
	values := make([]float64, population)
	for i := range remoras {
		remora := make([]float64, dimensions)
		for d := range remora {
			remora[d] = remoraRangeLow + rand.Float64()*(remoraRangeHigh-remoraRangeLow)
		}
		remoras[i] = remora
		values[i] = fn.Evaluate(remora)
	}

	best := append([]float64(nil), remoras[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), remoras[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range remoras {
			host := remoras[rand.IntN(population)]

			if rand.Float64() < 0.5 {
				for d := range remoras[i] {
					df := rand.Float64()*(best[d]-remoras[i][d]) + rand.Float64()*(host[d]-remoras[i][d])
					remoras[i][d] += df
				}
			} else {
				r1 := remoras[rand.IntN(population)]
				for d := range remoras[i] {
					remoras[i][d] += rand.Float64()*(best[d]-remoras[i][d]) + rand.Float64()*(r1[d]-remoras[i][d])
				}
			}

			for d := range remoras[i] {
				remoras[i][d] = clamp(remoras[i][d], remoraRangeLow, remoraRangeHigh)
			}

			values[i] = fn.Evaluate(remoras[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), remoras[i]...)
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
		Method:     "remora",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
