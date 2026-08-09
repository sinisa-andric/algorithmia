package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seaHorsePopulation = 20
	seaHorseMaxSteps   = 1000
	seaHorseTolerance  = 1e-6
	seaHorseRangeLow   = -5.0
	seaHorseRangeHigh  = 5.0
)

// SeaHorse minimizuje konfigurisanu benchmark funkciju koristeći Sea Horse Optimizer: morski konjic se ili
// spiralno kreće Levy letom oko najboljeg rešenja, ili ga direktno lovi (predacija) korakom koji opada tokom
// izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SeaHorse(problem models.Problem) (result models.Result, err error) {

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

	population := seaHorsePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seaHorseMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seaHorseTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	seahorses := make([][]float64, population)
	values := make([]float64, population)
	for i := range seahorses {
		seahorse := make([]float64, dimensions)
		for d := range seahorse {
			seahorse[d] = seaHorseRangeLow + rand.Float64()*(seaHorseRangeHigh-seaHorseRangeLow)
		}
		seahorses[i] = seahorse
		values[i] = fn.Evaluate(seahorse)
	}

	best := append([]float64(nil), seahorses[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), seahorses[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range seahorses {
			if rand.Float64() < 0.5 {
				l := mantegnaLevy(1.5)
				for d := range seahorses[i] {
					seahorses[i][d] = seahorses[i][d] + l*(best[d]-seahorses[i][d])
				}
			} else {
				for d := range seahorses[i] {
					seahorses[i][d] = seahorses[i][d] + rand.Float64()*(best[d]-seahorses[i][d])*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range seahorses[i] {
				seahorses[i][d] = clamp(seahorses[i][d], seaHorseRangeLow, seaHorseRangeHigh)
			}
			values[i] = fn.Evaluate(seahorses[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), seahorses[i]...)
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
		Method:     "sea_horse",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
