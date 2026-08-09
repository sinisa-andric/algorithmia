package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	fpsoPopulation = 20
	fpsoMaxSteps   = 500
	fpsoC1         = 2.0
	fpsoC2         = 2.0
	fpsoTolerance  = 1e-6
	fpsoRangeLow   = -5.0
	fpsoRangeHigh  = 5.0
	fpsoWindow     = 10
	fpsoWMin       = 0.4
	fpsoWMax       = 0.9
)

// FPSO minimizuje konfigurisanu benchmark funkciju koristeći Fuzzy Particle Swarm Optimization: inercija w se fazi
// pravilom prilagođava stopi napretka poslednjih koraka — spor napredak podiže w ka eksploraciji, brz napredak
// spušta w ka eksploataciji, a brzina svake čestice i dalje kombinuje sopstveno i globalno najbolje iskustvo
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func FPSO(problem models.Problem) (result models.Result, err error) {

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

	population := fpsoPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := fpsoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	c1 := fpsoC1
	if v, ok := problem.Payload["c1"].(float64); ok {
		c1 = v
	}

	c2 := fpsoC2
	if v, ok := problem.Payload["c2"].(float64); ok {
		c2 = v
	}

	tolerance := fpsoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, population)
	velocities := make([][]float64, population)
	values := make([]float64, population)
	personalBest := make([][]float64, population)
	personalBestValue := make([]float64, population)

	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = fpsoRangeLow + rand.Float64()*(fpsoRangeHigh-fpsoRangeLow)
		}
		positions[i] = position
		velocities[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(position)
		personalBest[i] = append([]float64(nil), position...)
		personalBestValue[i] = values[i]
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), positions[i]...)
		}
	}

	avgHistory := make([]float64, 0, fpsoWindow)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		avgValue := 0.0
		for _, v := range values {
			avgValue += v
		}
		avgValue /= float64(population)
		avgHistory = append(avgHistory, avgValue)
		if len(avgHistory) > fpsoWindow {
			avgHistory = avgHistory[1:]
		}

		improvementRate := 0.0
		if len(avgHistory) >= 2 {
			avgImprovement := (avgHistory[0] - avgHistory[len(avgHistory)-1]) / float64(len(avgHistory))
			improvementRate = avgImprovement / (avgHistory[len(avgHistory)-1] + 1e-10)
		}

		normalizedRate := clamp(improvementRate*10, 0, 1)
		w := fpsoWMax - (fpsoWMax-fpsoWMin)*normalizedRate

		for i := range positions {
			for d := range positions[i] {
				velocities[i][d] = w*velocities[i][d] + c1*rand.Float64()*(personalBest[i][d]-positions[i][d]) + c2*rand.Float64()*(best[d]-positions[i][d])
				positions[i][d] = clamp(positions[i][d]+velocities[i][d], fpsoRangeLow, fpsoRangeHigh)
			}
			values[i] = fn.Evaluate(positions[i])

			if values[i] < personalBestValue[i] {
				personalBestValue[i] = values[i]
				personalBest[i] = append([]float64(nil), positions[i]...)
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), positions[i]...)
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
		Method:     "fpso",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
