package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	horsePopulation = 20
	horseMaxSteps   = 1000
	horseTolerance  = 1e-6
	horseRangeLow   = -5.0
	horseRangeHigh  = 5.0
)

// Horse minimizuje konfigurisanu benchmark funkciju koristeći Horse Herd Optimization Algorithm: konj nasumično bira
// između ispaše (mala lokalna pretraga), imitacije predvodnika (kretanje ka najboljem konju) ili odbrane/bekstva
// (nasumičan skok amplitudom koja opada tokom izvršavanja)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Horse(problem models.Problem) (result models.Result, err error) {

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

	population := horsePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := horseMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := horseTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	horses := make([][]float64, population)
	values := make([]float64, population)
	for i := range horses {
		horse := make([]float64, dimensions)
		for d := range horse {
			horse[d] = horseRangeLow + rand.Float64()*(horseRangeHigh-horseRangeLow)
		}
		horses[i] = horse
		values[i] = fn.Evaluate(horse)
	}

	best := append([]float64(nil), horses[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), horses[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range horses {
			p := rand.Float64()
			switch {
			case p < 0.4:
				for d := range horses[i] {
					horses[i][d] = horses[i][d] + (rand.Float64()*2-1)*0.2
				}
			case p < 0.7:
				for d := range horses[i] {
					horses[i][d] = horses[i][d] + rand.Float64()*(best[d]-horses[i][d])
				}
			default:
				for d := range horses[i] {
					horses[i][d] = horses[i][d] + (rand.Float64()*2-1)*1.5*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range horses[i] {
				horses[i][d] = clamp(horses[i][d], horseRangeLow, horseRangeHigh)
			}
			values[i] = fn.Evaluate(horses[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), horses[i]...)
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
		Method:     "horse",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
