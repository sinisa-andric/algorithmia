package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seagullPopulation = 20
	seagullMaxSteps   = 1000
	seagullTolerance  = 1e-6
	seagullRangeLow   = -5.0
	seagullRangeHigh  = 5.0
)

// Seagull minimizuje konfigurisanu benchmark funkciju koristeći Seagull Optimization Algorithm: galeb prvo izbegava
// koliziju sa ostatkom jata skaliranim faktorom migracije koji opada tokom izvršavanja,
// zatim spiralno napada najbolje pronađeno rešenje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Seagull(problem models.Problem) (result models.Result, err error) {

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

	population := seagullPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seagullMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seagullTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	seagulls := make([][]float64, population)
	values := make([]float64, population)
	for i := range seagulls {
		seagull := make([]float64, dimensions)
		for d := range seagull {
			seagull[d] = seagullRangeLow + rand.Float64()*(seagullRangeHigh-seagullRangeLow)
		}
		seagulls[i] = seagull
		values[i] = fn.Evaluate(seagull)
	}

	best := append([]float64(nil), seagulls[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), seagulls[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		a := 2 - float64(steps)*(2/float64(maxSteps))

		for i := range seagulls {
			r := rand.Float64() * 2 * math.Pi
			x := r * math.Cos(r)
			y := r * math.Sin(r)

			for d := range seagulls[i] {
				cs := a * seagulls[i][d]
				ms := seagulls[i][d] + cs
				ds := math.Abs(ms - best[d])
				seagulls[i][d] = ds*x*y + best[d]
				seagulls[i][d] = clamp(seagulls[i][d], seagullRangeLow, seagullRangeHigh)
			}
			values[i] = fn.Evaluate(seagulls[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), seagulls[i]...)
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
		Method:     "seagull",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
