package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	arcticFoxPopulation = 20
	arcticFoxMaxSteps   = 1000
	arcticFoxTolerance  = 1e-6
	arcticFoxRangeLow   = -5.0
	arcticFoxRangeHigh  = 5.0
)

// ArcticFox minimizuje konfigurisanu benchmark funkciju koristeći Arctic Fox Optimization: lisica ili postepeno prati
// plen ka najboljem rešenju (sve izraženije tokom izvršavanja) ili simulira skok kroz sneg —
// nagli Levy skok u okolinu najboljeg rešenja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ArcticFox(problem models.Problem) (result models.Result, err error) {

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

	population := arcticFoxPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := arcticFoxMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := arcticFoxTolerance
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
			fox[d] = arcticFoxRangeLow + rand.Float64()*(arcticFoxRangeHigh-arcticFoxRangeLow)
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

		for i := range foxes {
			if rand.Float64() < 0.5 {
				for d := range foxes[i] {
					foxes[i][d] = foxes[i][d] + rand.Float64()*(best[d]-foxes[i][d])*(float64(steps)/float64(maxSteps))
				}
			} else {
				l := mantegnaLevy(1.5)
				for d := range foxes[i] {
					foxes[i][d] = best[d] + l*(rand.Float64()*2-1)
				}
			}

			for d := range foxes[i] {
				foxes[i][d] = clamp(foxes[i][d], arcticFoxRangeLow, arcticFoxRangeHigh)
			}
			values[i] = fn.Evaluate(foxes[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), foxes[i]...)
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
		Method:     "arctic_fox",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
