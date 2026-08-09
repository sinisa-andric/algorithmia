package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seaUrchinPopulation = 20
	seaUrchinMaxSteps   = 1000
	seaUrchinTolerance  = 1e-6
	seaUrchinRangeLow   = -5.0
	seaUrchinRangeHigh  = 5.0
)

// SeaUrchin minimizuje konfigurisanu benchmark funkciju koristeći Sea Urchin Optimizer: morski jež najčešće vrši
// sporu bodljikavu lokalnu pretragu oko trenutne pozicije uz slabu vezu ka najboljem rešenju, a povremeno se
// kotrlja direktno ka najboljem
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SeaUrchin(problem models.Problem) (result models.Result, err error) {

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

	population := seaUrchinPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seaUrchinMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seaUrchinTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	urchins := make([][]float64, population)
	values := make([]float64, population)
	for i := range urchins {
		urchin := make([]float64, dimensions)
		for d := range urchin {
			urchin[d] = seaUrchinRangeLow + rand.Float64()*(seaUrchinRangeHigh-seaUrchinRangeLow)
		}
		urchins[i] = urchin
		values[i] = fn.Evaluate(urchin)
	}

	best := append([]float64(nil), urchins[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), urchins[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range urchins {
			if rand.Float64() < 0.8 {
				for d := range urchins[i] {
					urchins[i][d] = urchins[i][d] + (rand.Float64()*2-1)*0.2 + rand.Float64()*(best[d]-urchins[i][d])*0.3
				}
			} else {
				for d := range urchins[i] {
					urchins[i][d] = urchins[i][d] + rand.Float64()*(best[d]-urchins[i][d])
				}
			}

			for d := range urchins[i] {
				urchins[i][d] = clamp(urchins[i][d], seaUrchinRangeLow, seaUrchinRangeHigh)
			}
			values[i] = fn.Evaluate(urchins[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), urchins[i]...)
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
		Method:     "sea_urchin",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
