package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sootyTernPopulation = 20
	sootyTernMaxSteps   = 1000
	sootyTernTolerance  = 1e-6
	sootyTernRangeLow   = -5.0
	sootyTernRangeHigh  = 5.0
)

// SootyTern minimizuje konfigurisanu benchmark funkciju koristeći Sooty Tern Optimization Algorithm: čvorak izbegava
// koliziju sa jatom faktorom konvergencije koji opada tokom izvršavanja, zatim spiralno napada najbolje pronađeno
// rešenje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SootyTern(problem models.Problem) (result models.Result, err error) {

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

	population := sootyTernPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sootyTernMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sootyTernTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	terns := make([][]float64, population)
	values := make([]float64, population)
	for i := range terns {
		tern := make([]float64, dimensions)
		for d := range tern {
			tern[d] = sootyTernRangeLow + rand.Float64()*(sootyTernRangeHigh-sootyTernRangeLow)
		}
		terns[i] = tern
		values[i] = fn.Evaluate(tern)
	}

	best := append([]float64(nil), terns[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), terns[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		cf := 2 * rand.Float64() * (1 - float64(steps)/float64(maxSteps))

		for i := range terns {
			r := rand.Float64() * 2 * math.Pi
			for d := range terns[i] {
				sa := cf * terns[i][d]
				sb := sa + terns[i][d]
				sc := math.Abs(sb - best[d])
				terns[i][d] = sc*math.Sin(r)*math.Cos(r)*rand.Float64() + best[d]
				terns[i][d] = clamp(terns[i][d], sootyTernRangeLow, sootyTernRangeHigh)
			}
			values[i] = fn.Evaluate(terns[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), terns[i]...)
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
		Method:     "sooty_tern",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
