package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	fennecFoxPopulation = 20
	fennecFoxMaxSteps   = 1000
	fennecFoxTolerance  = 1e-6
	fennecFoxRangeLow   = -5.0
	fennecFoxRangeHigh  = 5.0
)

// FennecFox minimizuje konfigurisanu benchmark funkciju koristeći Fennec Fox Optimization: svaki lisac ili traži
// hranu krećući se duž razlike dva nasumična lisca, ili beži od pretnje ka trenutno najboljem rešenju,
// oba koraka skalirana temperaturom koja eksponencijalno opada
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func FennecFox(problem models.Problem) (result models.Result, err error) {

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

	population := fennecFoxPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := fennecFoxMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := fennecFoxTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	foxes := make([][]float64, population)
	values := make([]float64, population)
	for i := range foxes {
		fox := make([]float64, dimensions)
		for d := range fox {
			fox[d] = fennecFoxRangeLow + rand.Float64()*(fennecFoxRangeHigh-fennecFoxRangeLow)
		}
		foxes[i] = fox
		values[i] = fn.Evaluate(fox)
	}

	best := append([]float64(nil), foxes[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), foxes[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		t := math.Exp(-float64(steps) / float64(maxSteps))

		for i := range foxes {
			if rand.Float64() < 0.5 {
				r1 := foxes[rand.IntN(population)]
				r2 := foxes[rand.IntN(population)]
				for d := range foxes[i] {
					foxes[i][d] += t * (r1[d] - r2[d])
				}
			} else {
				for d := range foxes[i] {
					foxes[i][d] = best[d] + t*rand.Float64()*(best[d]-foxes[i][d])
				}
			}

			for d := range foxes[i] {
				foxes[i][d] = clamp(foxes[i][d], fennecFoxRangeLow, fennecFoxRangeHigh)
			}

			values[i] = fn.Evaluate(foxes[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), foxes[i]...)
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
		Method:     "fennec_fox",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
