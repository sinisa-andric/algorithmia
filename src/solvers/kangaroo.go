package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	kangarooPopulation = 20
	kangarooMaxSteps   = 1000
	kangarooTolerance  = 1e-6
	kangarooRangeLow   = -5.0
	kangarooRangeHigh  = 5.0
)

// Kangaroo minimizuje konfigurisanu benchmark funkciju koristeći Kangaroo Mob Optimization: klokani lošiji od proseka
// grupe rade veliki skok ka najboljem rešenju amplitudom koja opada tokom izvršavanja,
// dok klokani bolji od proseka rade mali lokalni skok
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Kangaroo(problem models.Problem) (result models.Result, err error) {

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

	population := kangarooPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := kangarooMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := kangarooTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	kangaroos := make([][]float64, population)
	values := make([]float64, population)
	for i := range kangaroos {
		kangaroo := make([]float64, dimensions)
		for d := range kangaroo {
			kangaroo[d] = kangarooRangeLow + rand.Float64()*(kangarooRangeHigh-kangarooRangeLow)
		}
		kangaroos[i] = kangaroo
		values[i] = fn.Evaluate(kangaroo)
	}

	best := append([]float64(nil), kangaroos[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), kangaroos[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		jumpPower := 2 * (1 - float64(steps)/float64(maxSteps))

		avgValue := 0.0
		for _, v := range values {
			avgValue += v
		}
		avgValue /= float64(population)

		for i := range kangaroos {
			if values[i] > avgValue {
				for d := range kangaroos[i] {
					kangaroos[i][d] = kangaroos[i][d] + jumpPower*rand.Float64()*(best[d]-kangaroos[i][d])
				}
			} else {
				for d := range kangaroos[i] {
					kangaroos[i][d] = kangaroos[i][d] + (rand.Float64()*2-1)*0.2
				}
			}

			for d := range kangaroos[i] {
				kangaroos[i][d] = clamp(kangaroos[i][d], kangarooRangeLow, kangarooRangeHigh)
			}
			values[i] = fn.Evaluate(kangaroos[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), kangaroos[i]...)
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
		Method:     "kangaroo",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
